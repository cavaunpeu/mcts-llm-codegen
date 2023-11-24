import asyncio
import itertools

import wandb

from config import experiments
from mcts import stub, MCTS
from type import APPSProblem
from util import compute_reward, parse_args


def compute_test_reward(code, problem_index):
    return compute_reward(
        code,
        APPSProblem(problem_index),
        mode="test",
    )


async def run(args):
    # Initialize
    exp = experiments[args.experiment_name]
    configs = list(itertools.product(*exp["param_ranges"].values()))
    model_paths = exp["model_paths"]
    # Run MCTS
    futures, results = [], []
    for idx in APPSProblem.problem_indices:
        for model_path in model_paths:
            mcts = MCTS(
                args.debug,
                args.dry,
                model_path=model_path,
            )
            if args.remote:
                f = mcts.run.remote.aio
                futures += [f(*(cfg + (idx,))) for cfg in configs]
            else:
                f = mcts.run.local
                results += [f(*(cfg + (idx,))) for cfg in configs]

    # Collect results
    iterable = asyncio.as_completed(futures) if args.remote else results
    for payload in iterable:
        payload = await payload if args.remote else payload
        if not args.dry:
            result, config = payload["result"], payload["config"]  # type: ignore
            code, problem_index = result["code"], config["problem_index"]
            test_reward = compute_test_reward(code, problem_index)
            # Save to wandb
            wandb.init(
                group=args.experiment_name,
                config=config,
            )
            wandb.log({**result, "test_reward": test_reward})
            wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    if not args.dry and args.experiment_name is None:
        raise ValueError("Must specify experiment name")
    with stub.run():
        asyncio.run(run(args))
