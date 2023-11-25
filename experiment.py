import asyncio
import itertools

import wandb

from config import experiments
from mcts import stub, MCTS
from type import APPSProblem
from util import compute_reward, filter_experiment_configs, parse_args


def compute_test_reward(code, problem_index):
    return compute_reward(
        code,
        APPSProblem(problem_index),
        mode="test",
    )


async def run(args):
    # Initialize experiment
    exp = experiments[args.experiment_name]
    configs = list(itertools.product(*exp.values()))
    configs = [dict(zip(exp.keys(), cfg)) for cfg in configs]
    # Get existing experiment results
    configs = filter_experiment_configs(configs, args.experiment_name)
    print(f"Running {len(configs)} configs ...")
    # Run MCTS
    func_name = "remote.aio" if args.remote else "local"
    results = []
    for idx in APPSProblem.problem_indices:
        for cfg in configs:
            mcts = MCTS(
                args.debug,
                args.dry,
                model_path=cfg["model_path"],
            )
            f = getattr(mcts.run, func_name)
            result = f(**{**cfg, "problem_index": idx})
            results.append(result)

    # Collect results
    iterable = asyncio.as_completed(results) if args.remote else results
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
