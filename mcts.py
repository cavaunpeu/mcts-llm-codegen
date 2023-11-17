import os
from time import time

import modal

from const import TEST_PROBLEMS_DIR
from type import ModelContext, Node, Policy, Problem
from util import compute_reward, extract_code, log_info

# Suppress noisy warnings from reward evaluation code
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Modal config
stub = modal.Stub(
    "mcts-llm-codegen",
    image=modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install("transformers", "gdown", "pyext")
    .run_commands(
        # Download 1.5B param model
        "gdown 1svUcwtqL6AD_Ti0eXJS03AaMdS7HDZ0d -O /root/",
        # Extract model
        "mkdir -p /root/models",
        "tar -xvf /root/models_1.5B.tar -C /root/models",
    ),
)


@stub.cls(
    gpu="any",
    secret=modal.Secret.from_dict(
        {"TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"]}
    ),
    container_idle_timeout=60 * 2,
    mounts=[
        modal.Mount.from_local_dir(
            TEST_PROBLEMS_DIR, remote_path=f"/root/{TEST_PROBLEMS_DIR}"
        )
    ],
)
class MCTS:
    def __init__(
        self,
        test_problem_index: int,
        debug: bool,
        dry: bool,
    ):
        self.problem = Problem(TEST_PROBLEMS_DIR, test_problem_index)
        self.policy = Policy()
        self.debug = debug
        self.dry = dry

    def __enter__(self):
        if not self.dry:
            self.ctx = ModelContext()
            self.ctx.initialize()
            self.tokenizer = self.ctx.tokenizer

    @modal.method()
    def run(self, k: int, num_rollouts: int):
        """
        Run MCTS on the given problem.

        Args:
            k: Number of expanded children.

        Returns:
            (code, reward): Generated code and reward.
        """
        if self.dry:
            return ("dummy code", 1)
        state = self.tokenizer.encode(self.problem.prompt)
        node = root = Node(
            state=state,
            action=None,
            prob=None,
            parent=None,
            model_context=self.ctx,
            k=k,
        )
        num_actions = 0
        total_elapsed = 0
        rewards_cache = {}
        result = list(state)
        while True:
            start = time()
            # Perform rollouts
            for i in range(1, num_rollouts + 1):
                # Start at root
                node = root
                # Selection (select a leaf node)
                while True:
                    if node.is_leaf_node:
                        break
                    node = max(node.children, key=self.policy)
                if not node.state[-1] == self.ctx.terminal_token_id:
                    # Expansion (expand children, select one to rollout)
                    # NB: If scores are the same, first node will always be selected.  # noqa: E501
                    node = max(node.children, key=self.policy)
                    # Simulate (simulate rollout)
                    output = self.ctx.generate(node.state)  # noqa: E501
                    text = self.tokenizer.decode(output["sequence"])
                    code = extract_code(text)
                    # Compute reward
                    key = (code, self.problem)
                    if key not in rewards_cache:
                        rewards_cache[key] = compute_reward(code, self.problem)
                    reward = rewards_cache[key]
                    # Backpropagation (update node statistics)
                    while node:
                        node.visits += 1
                        node.observed_rewards.append(reward)
                        node = node.parent
            # Take action, reset root
            node = root = max(root.children, key=lambda node: node.value)
            # Log action
            elapsed = time() - start
            if self.debug:
                log_info(
                    num_actions,
                    node,
                    None,
                    self.tokenizer.decode(node.action),
                    elapsed,
                )
            result.append(node.action)
            num_actions += 1
            total_elapsed += elapsed
            # Check if we're done
            if node.state[-1] == self.ctx.terminal_token_id:
                break
        code = extract_code(self.tokenizer.decode(result))
        reward = compute_reward(code, self.problem)
        if self.debug:
            print(f"\n>>> Result:\n\n{code}")
            print(
                f"\n>>> Reward: {reward} | Elapsed: {total_elapsed:.3f}s | Generations: {self.ctx.generations}"  # noqa: E501
            )
        return (code, reward)