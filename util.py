import argparse
import re
import sys
from typing import Optional
from graphviz import Digraph

import numpy as np
import wandb

from type import APPSProblem, Node
from const import (
    CONCURRENCY_LIMIT,
    K,
    NO_CUDA,
    NUM_ROLLOUTS,
    TERMINAL_TOKEN,
    PROBLEM_INDEX,
)

sys.path.append("Code-AI-Tree-Search/eval")
from compute_reward import compute_reward as _compute_reward  # type: ignore


def extract_code(text: str, terminal_token: str = TERMINAL_TOKEN) -> str:
    pattern = rf"ANSWER:\n(.*?){re.escape(terminal_token)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def compute_reward(code: str, problem: APPSProblem, mode: str = "train") -> int:
    return _compute_reward(
        problem.dir, code + "\n", mode=mode, public_test_cases="half"
    )


def log_info(
    num_actions: int,
    node: Node,
    token: Optional[str],
    elapsed: Optional[float],
):
    print(
        f"Step: {('Prediction' if elapsed is not None else 'Selection'):<10} |",
        f"Action #: {num_actions:<2} |",
        f"Action: {node.display_action if node else 'N/A':<6} |",
        f"Token: {repr(token) if token is not None else 'N/A':<8} |",
        f"Elapsed: {(str(np.round(elapsed, 3)) + 's' if elapsed is not None else 'N/A'):<7} |",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", default=False)
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument("--dry", action="store_true", default=False)
    parser.add_argument("--no_cuda", action="store_true", default=NO_CUDA)
    parser.add_argument("--K", type=int, help="Number of expanded children", default=K)
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument("--concurrency_limit", type=int, default=CONCURRENCY_LIMIT)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument(
        "--problem_index",
        type=str,
        default=PROBLEM_INDEX,
        choices=APPSProblem.problem_indices,
    )
    args, _ = parser.parse_known_args()
    return args


def traverse_and_visualize(node, graph, tokenizer, node_id=0):
    if node is None:
        return node_id

    # Create a label for the node with its statistics
    action = tokenizer.decode([node.action]) if node.action != "root" else "root"
    label = f"Action ID: {node.action}\n Action: {action}\nVisits: {node.visits}\nSelected: {node.selected}"
    if node.action != "root":
        label += f"\nProb: {node.prob:.2f}\nValue: {node.value:.2f}"
    graph.node(str(node_id), label)

    current_id = node_id
    children = [c for c in node.children if c.selected > 0 or c.visits > 0]
    for child in children:
        next_id = node_id + 1
        graph.edge(str(current_id), str(next_id))
        node_id = traverse_and_visualize(child, graph, tokenizer, next_id)

    return node_id


def visualize_tree(root, tokenizer):
    graph = Digraph(comment="Tree Visualization")
    traverse_and_visualize(root, graph, tokenizer)
    graph.render("tree", format="png")


def filter_experiment_configs(
    configs, experiment_name, project_name="mcts-llm-codegen"
):
    wandb_api = wandb.Api()
    runs = wandb_api.runs(project_name, filters={"group": experiment_name})
    already_run = [run.config for run in runs]
    return [cfg for cfg in configs if cfg not in already_run]
