import os
from time import time
from typing import Dict

import modal

from const import TEST_PROBLEMS_DIR
from type import ModelContext, Node, Policy, Problem
from util import compute_reward, extract_code, log_info, parse_args, visualize_tree

# Suppress noisy warnings from reward evaluation code
os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = parse_args()

# Modal config
stub = modal.Stub(
    image=(
        modal.Image.debian_slim()
        if args.dry or args.no_cuda
        else modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    )
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
    gpu="any"
    if args.remote and (not args.no_cuda) and (not args.dry)
    else None,  # noqa: E501,
    secret=modal.Secret.from_dict(
        {"TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"]}
    ),
    container_idle_timeout=60 * 2,
    mounts=[
        modal.Mount.from_local_dir(
            TEST_PROBLEMS_DIR, remote_path=f"/root/{TEST_PROBLEMS_DIR}"
        )
    ],
    concurrency_limit=args.concurrency_limit,
    timeout=60 * 60,
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
        start_time = time()
        if self.dry:
            return {
                "code": "dummy code",
                "reward": 1,
                "start_time": start_time,
                "elapsed_ms": num_rollouts / 100,
            }
        state = self.tokenizer.encode(self.problem.prompt)
        node = root = absolute_root = Node(
            state=state,
            action="root",
            prob=None,
            parent=None,
            model_context=self.ctx,
            k=k,
        )
        num_actions = 0
        total_elapsed = 0
        rewards_cache = {}
        result = list(state)
        while len(result) < self.ctx.max_gen_horizon:
            start = time()
            # Perform rollouts
            for _ in range(1, num_rollouts + 1):
                # Start at root
                node = root
                # Selection (select a leaf node)
                while True:
                    if node.is_leaf_node:
                        node.selected += 1
                        break
                    node = max(node.children, key=self.policy)
                # Expansion (expand children, select one to rollout)
                if node.action != self.ctx.terminal_token_id:
                    # NB: If scores are the same, first child will always be selected.  # noqa: E501
                    node = max(node.children, key=self.policy)
                # Simulate (simulate rollout)
                output = self.ctx.generate(node.state)  # noqa: E501
                text = self.tokenizer.decode(output["sequence"])
                code = extract_code(text)
                # Compute reward
                if code not in rewards_cache:
                    rewards_cache[code] = compute_reward(
                        code, self.problem
                    )  # noqa: E501
                reward = rewards_cache[code]
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
                    self.tokenizer.decode(node.action),
                    elapsed,
                )
            result.append(node.action)
            num_actions += 1
            total_elapsed += elapsed
            # Check if we're done
            if node.action == self.ctx.terminal_token_id:
                break
        code = max(rewards_cache, key=rewards_cache.get)
        reward = rewards_cache[code]
        if self.debug:
            visualize_tree(absolute_root, self.ctx.tokenizer)
        return {
            "code": code,
            "reward": reward,
            "start_time": start_time,
            "elapsed_ms": total_elapsed,
            "num_sequence_generations": self.ctx.num_sequence_gens,
            "num_next_token_generations": self.ctx.num_next_token_gens,
            "num_unique_program_generations": len(rewards_cache),
        }
