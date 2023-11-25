import asyncio

import wandb

from mcts import stub, MCTS
from type import APPSProblem
from util import compose_configs, compute_reward, parse_args


def compute_test_reward(code, problem_index):
    return compute_reward(
        code,
        APPSProblem(problem_index),
        mode="test",
    )


async def run(args):
    # Compose experiment configs
    configs = compose_configs(
        APPSProblem.problem_indices, args.experiment_name, args.dry
    )
    print(f"Running {len(configs)} remaining configs ...")
    # Run MCTS
    results = []
    for cfg in configs:
        mcts = MCTS(
            args.debug * args.concurrency_limit == 1,
            args.dry,
            model_path=cfg["model_path"],
        )
        f = mcts.run.remote.aio if args.remote else mcts.run.local
        results.append(f(**cfg))

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
