import os
from time import time

import modal
from const import MODEL_PATH

from stub import stub
from type import ModelContext, Node, Policy, APPSProblem
from util import compute_reward, extract_code, log_info, parse_args, visualize_tree

# Suppress noisy warnings from reward evaluation code
os.environ["TOKENIZERS_PARALLELISM"] = "false"
args = parse_args()


@stub.cls(
    gpu="any" if args.remote and (not args.no_cuda) and (not args.dry) else None,
    cloud="aws",
    secret=modal.Secret.from_dict(
        {"TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"]}
    ),
    container_idle_timeout=60 * 2,
    mounts=[
        modal.Mount.from_local_dir(
            APPSProblem.base_path, remote_path=f"/root/{APPSProblem.base_path}"
        )
    ],
    concurrency_limit=args.concurrency_limit,
    timeout=60 * 60,
)
class MCTS:
    def __init__(
        self,
        debug: bool,
        dry: bool,
        model_path: str = MODEL_PATH,
    ):
        self.debug = debug
        self.dry = dry
        self.model_path = model_path

    def __enter__(self):
        if not self.dry:
            self.policy = Policy()
            self.ctx = ModelContext(self.model_path)
            self.ctx.initialize()
            self.tokenizer = self.ctx.tokenizer

    @modal.method()
    def run(self, k: int, num_rollouts: int, problem_index: str, **kwargs):
        """
        Run MCTS on the given problem.

        Args:
            k: Number of expanded children.

        Returns:
            (code, reward): Generated code and reward.
        """
        # Initialize
        config = {k: v for k, v in locals().copy().items() if k != "self"}
        start_time = time()
        if self.dry:
            return {
                "config": config,
                "result": {
                    "code": "dummy code",
                    "train_reward": 1,
                    "start_time": start_time,
                },
            }
        # Run MCTS
        problem = APPSProblem(problem_index)
        state = self.tokenizer.encode(problem.prompt)
        stats = {
            "num_sequence_gens": 0,
            "num_next_token_gens": 0,
            "generation_times": [],
        }
        num_actions = 0
        total_elapsed = 0
        rewards_cache = {}
        result = list(state)
        # Define root node
        node = root = absolute_root = Node(
            state=state,
            action="root",
            prob=None,
            parent=None,
            model_context=self.ctx,
            k=k,
            stats=stats,
        )
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
                    # NB: If scores are the same, first child will always be selected.
                    node = max(node.children, key=self.policy)
                # Simulate (simulate rollout)
                output = self.ctx.generate(node.state, stats)
                text = self.tokenizer.decode(output["sequence"])
                code = extract_code(text)
                # Compute reward
                if code not in rewards_cache:
                    rewards_cache[code] = compute_reward(code, problem)
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
        code = max(rewards_cache, key=rewards_cache.get)  # type: ignore
        reward = rewards_cache[code]
        if self.debug:
            visualize_tree(absolute_root, self.ctx.tokenizer)
        return {
            "config": config,
            "result": {
                "code": code,
                "train_reward": reward,
                "train_reward_from_selected_nodes_program": compute_reward(
                    extract_code(self.tokenizer.decode(result)), problem
                ),
                "start_time": start_time,
                "elapsed_ms": total_elapsed,
                "generation_elapsed_ms": sum(stats["generation_times"]),
                "num_sequence_generations": stats["num_sequence_gens"],
                "num_next_token_generations": stats["num_next_token_gens"],
                "num_unique_program_generations": len(rewards_cache),
            },
        }
