import argparse
import os
import re
import sys
from typing import Optional

import numpy as np

from type import Node, Problem
from const import (
    CONCURRENCY_LIMIT,
    K,
    NO_CUDA,
    NUM_ROLLOUTS,
    TERMINAL_TOKEN,
    TEST_PROBLEM_INDEX,
    TEST_PROBLEMS_DIR,
)  # noqa: E501

sys.path.append("Code-AI-Tree-Search/eval")
from compute_reward import compute_reward as _compute_reward  # type: ignore


def extract_code(text: str, terminal_token: str = TERMINAL_TOKEN) -> str:
    pattern = rf"ANSWER:\n(.*?){re.escape(terminal_token)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def compute_reward(code: str, problem: Problem) -> int:
    return _compute_reward(
        problem.dir, code + "\n", mode="train", public_test_cases="half"
    )


def log_info(
    num_actions: int,
    node: Node,
    rollout_index: Optional[int],
    token: Optional[str],
    elapsed: Optional[float],
):
    print(
        f"Step: {('Prediction' if elapsed is not None else 'Selection'):<10} |",  # noqa: E501
        f"Action #: {num_actions:<2} |",
        f"Rollout #: {rollout_index if rollout_index is not None else 'N/A':<4} |",  # noqa: E501
        f"Action: {node.display_action if node else 'N/A':<6} |",
        f"Token: {repr(token) if token is not None else 'N/A':<8} |",
        f"Elapsed: {(str(np.round(elapsed, 3)) + 's' if elapsed is not None else 'N/A'):<7} |",  # noqa: E501
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument("--no_cuda", action="store_true", default=NO_CUDA)
    parser.add_argument(
        "--K", type=int, help="Number of expanded children", default=K
    )  # noqa: E501
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument(
        "--concurrency_limit", type=int, default=CONCURRENCY_LIMIT
    )  # noqa: E501
    parser.add_argument(
        "--test_problem_index",
        type=str,
        default=TEST_PROBLEM_INDEX,
        choices=os.listdir(TEST_PROBLEMS_DIR),
    )  # noqa: E501
    args, _ = parser.parse_known_args()
    return args
