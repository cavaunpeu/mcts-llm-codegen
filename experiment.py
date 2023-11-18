import itertools
from pprint import pprint
from mcts import stub, MCTS
from util import parse_args


PARAM_RANGES = {
    "K": [1, 2],
    "num_rollouts": [1, 2],
}
PARAMS = list(itertools.product(*PARAM_RANGES.values()))


if __name__ == "__main__":
    args = parse_args()
    with stub.run():
        print(f"Running MCTS on test problem {args.test_problem_index}...")
        mcts = MCTS(
            args.test_problem_index,
            args.debug,
            args.dry,
        )
        output = mcts.run.starmap(PARAMS)
        results = [
            {**res, "params": dict(zip(PARAM_RANGES.keys(), vals))}
            for res, vals in zip(output, PARAMS)
        ]
        pprint(results)
