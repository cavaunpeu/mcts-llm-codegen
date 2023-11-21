import asyncio
import itertools
from pprint import pprint

from const import MODEL_PATH
from mcts import stub, MCTS
from type import APPSProblem
from util import compute_reward, parse_args


PARAM_RANGES = {"K": [3], "num_rollouts": range(1, 3)}
MODELS_PATHS = [MODEL_PATH]
PARAMS = list(itertools.product(*PARAM_RANGES.values()))


async def run(args, params=PARAMS):
    handlers = []
    for model_path in MODELS_PATHS:
        print(f"Running MCTS on problem {args.problem_index}...")
        mcts = MCTS(
            args.problem_index,
            args.debug,
            args.dry,
            model_path=model_path,
        )
        func = mcts.run.remote.aio if args.remote else mcts.run.local
        handlers += [func(*prm) for prm in params]
    return await asyncio.gather(*handlers), params


def enrich_results(results, params):
    return [
        {
            **res,
            "params": dict(zip(PARAM_RANGES.keys(), prm)),
            "test_reward": (
                compute_reward(
                    res["code"],
                    APPSProblem(res["problem_index"]),
                    mode="test",
                )
                if not args.dry
                else None
            ),
        }
        for res, prm in zip(results, params)
    ]


if __name__ == "__main__":
    args = parse_args()
    with stub.run():
        results, params = asyncio.run(run(args))
        results = enrich_results(results, params)
        pprint(results)
