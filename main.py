from mcts import stub, MCTS
from util import parse_args


if __name__ == "__main__":
    args = parse_args()
    params = {
        "k": args.K,
        "num_rollouts": args.num_rollouts,
        "problem_index": args.problem_index,
    }
    with stub.run():
        print(f"Running MCTS on problem {args.problem_index}...")
        mcts = MCTS(
            args.debug,
            args.dry,
        )
        output = mcts.run.remote(**params) if args.remote else mcts.run.local(**params)
        for key, val in output.items():
            print(f"{key}: {val}")
