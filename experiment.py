from mcts import stub, MCTS
from util import parse_args


PARAMS = [
    {"k": 1, "num_rollouts": 1},
    {"k": 1, "num_rollouts": 2},
    {"k": 1, "num_rollouts": 3},
    {"k": 2, "num_rollouts": 1},
    {"k": 2, "num_rollouts": 2},
    {"k": 2, "num_rollouts": 3},
    {"k": 3, "num_rollouts": 1},
    {"k": 3, "num_rollouts": 2},
    {"k": 3, "num_rollouts": 3},
]


if __name__ == "__main__":
    args = parse_args()
    with stub.run():
        print(f"Running MCTS on test problem {args.test_problem_index}...")
        mcts = MCTS(
            args.test_problem_index,
            args.debug,
            args.dry,
        )
        results = list(mcts.run.starmap(PARAMS))
        print(results)
