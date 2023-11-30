import asyncio
from app.mcts import stub, MCTS
from app.type import APPSProblem
from app.util import compose_configs, parse_args


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
            args.experiment_name,
            model_path=cfg["model_path"],
        )
        f = mcts.run.remote.aio if args.remote else mcts.run.local
        results.append(f(**cfg))

    # Collect results
    if args.remote:
        await asyncio.gather(*results)


if __name__ == "__main__":
    args = parse_args()
    if not args.dry and args.experiment_name is None:
        raise ValueError("Must specify experiment name")
    with stub.run(detach=args.remote):
        asyncio.run(run(args))
