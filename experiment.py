from mcts import stub, MCTS
from util import parse_args


K = [1, 2, 3]
NUM_ROLLOUTS = [1, 2, 3]
PARAMS = [(k, nr) for k in K for nr in NUM_ROLLOUTS]


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
