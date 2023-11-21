import itertools
from pprint import pprint
from mcts import stub, MCTS
from type import Problem
from util import compute_reward, parse_args


PARAM_RANGES = {
    "K": [3],
    "num_rollouts": range(1, 4),
}
PARAMS = list(itertools.product(*PARAM_RANGES.values()))


if __name__ == "__main__":
    args = parse_args()
    with stub.run():
        print(f"Running MCTS on problem {args.problem_index}...")
        mcts = MCTS(
            args.problem_index,
            args.debug,
            args.dry,
        )
        output = mcts.run.starmap(PARAMS)
        results = [
            {
                **res,
                "params": dict(zip(PARAM_RANGES.keys(), vals)),
                "test_reward": compute_reward(
                    res["code"], APPSProblem(res["problem_index"]), mode="test"
                )
                if not args.dry
                else None,
            }
            for res, vals in zip(output, PARAMS)
        ]
        pprint(results)
