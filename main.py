import argparse
import os
from time import time

import modal

from const import NUM_ROLLOUTS, TEST_PROBLEM_INDEX, TEST_PROBLEMS_DIR, K
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
        k: int,
        test_problem_index: int,
        num_rollouts: int,
        debug: bool,
    ):
        self.problem = Problem(TEST_PROBLEMS_DIR, test_problem_index)
        self.policy = Policy()
        self.k = k
        self.num_rollouts = num_rollouts
        self.debug = debug

    def __enter__(self):
        self.ctx = ModelContext(self.k)
        self.ctx.initialize()
        self.tokenizer = self.ctx.tokenizer

    @modal.method()
    def run(self):
        state = self.tokenizer.encode(self.problem.prompt)
        node = root = Node(
            state=state,
            action=None,
            prob=None,
            parent=None,
            model_context=self.ctx,
        )
        num_actions = 0
        total_elapsed = 0
        rewards_cache = {}
        result = list(state)
        while True:
            start = time()
            # Perform rollouts
            for i in range(1, self.num_rollouts + 1):
                # Start at root
                node = root
                # Selection (select a leaf node)
                while True:
                    if node.is_leaf_node:
                        if self.debug:
                            log_info(num_actions, node, i, None, None)
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
        print(f"\n>>> Result:\n\n{code}")
        print(
            f"\n>>> Reward: {reward} | Elapsed: {total_elapsed:.3f}s | Generations: {self.ctx.generations}"  # noqa: E501
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument(
        "--K", type=int, help="Number of expanded children", default=K
    )  # noqa: E501
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument(
        "--test_problem_index",
        type=str,
        default=TEST_PROBLEM_INDEX,
        choices=os.listdir(TEST_PROBLEMS_DIR),
    )  # noqa: E501
    args = parser.parse_args()
    with stub.run():
        print(f"Running MCTS on test problem {args.test_problem_index}...")
        mcts = MCTS(
            args.K,
            args.test_problem_index,
            args.num_rollouts,
            args.debug,
        )
        mcts.run.remote() if args.remote else mcts.run.local()
