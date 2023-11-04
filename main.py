import argparse
from collections import defaultdict
from dataclasses import dataclass
import re
from time import time
from typing import List, Optional, Union
import numpy as np
import torch
import transformers

import sys

sys.path.append("Code-AI-Tree-Search/eval")
from compute_reward import compute_reward as _compute_reward  # type: ignore


MODEL_NAME = "gpt2"
MODEL_PATH = "models/1.5B"
TEST_PROBLEMS_DIR = "APPS/test"
MAX_GEN_HORIZON = 1024
K = 3
TEST_PROBLEM_INDEX = 4136
NO_CUDA = False
TERMINAL_TOKEN = "<|endoftext|>"
SEED = 1
UCB_BASE = 10
UCB_CONSTANT = 4
NUM_ROLLOUTS = 3
NUM_BEAMS = 1


class Problem:
    def __init__(self, base_path: str, idx: int):
        self.dir = f"{base_path}/{idx}"
        self.input_output_path = f"{self.dir}/input_output.json"
        self.question_path = f"{self.dir}/question.txt"
        self.solutions_path = f"{self.dir}/solutions.json"
        self._prompt = None

    @property
    def prompt(self):
        if self._prompt is None:
            self._prompt = self._generate_prompt()
        return self._prompt

    def _generate_prompt(self):
        prompt = "\nQUESTION:\n"
        with open(self.question_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            prompt += data
        prompt += "\nUse Standard Input format"
        prompt += "\nANSWER:\n"
        return prompt


# add test that makes prompt equal to: '\nQUESTION:\nA + B is often used as an example of the easiest problem possible to show some contest platform. However, some scientists have observed that sometimes this problem is not so easy to get accepted. Want to try?\n\n\n-----Input-----\n\nThe input contains two integers a and b (0 ≤ a, b ≤ 10^3), separated by a single space.\n\n\n-----Output-----\n\nOutput the sum of the given integers.\n\n\n-----Examples-----\nInput\n5 14\n\nOutput\n19\n\nInput\n381 492\n\nOutput\n873\nUse Standard Input format\nANSWER:\n'


@dataclass
class OutputTrieNode:
    id: int
    score_tensor: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.children = {}


class OutputTrie:
    def __init__(self, terminal_token_id: int):
        self.terminal_token_id = terminal_token_id
        self.root = OutputTrieNode(id=-1)

    def insert(self, sequence: List[int], scores: List[torch.Tensor]):
        input_seq_len = len(sequence) - len(scores)
        scores = [None] * input_seq_len + scores  # type: ignore
        node = self.root
        for id, tensor in zip(sequence, scores):
            if id not in node.children:
                node.children[id] = OutputTrieNode(id, tensor)
            node = node.children[id]

    def search(self, sequence: List[int], next_token_only: bool = False):
        node = self.root
        output = defaultdict(list)
        # Navigate to input sequence tip
        for id in sequence:
            if id not in node.children:
                return
            node = node.children[id]
            output["sequence"].append(id)
        # Walk remaining output sequence
        while len(node.children) == 1:
            (id,) = list(node.children)
            node = node.children[id]
            output["sequence"].append(id)
            output["scores"].append(node.score_tensor)
            if next_token_only:
                return output
        if node.id == self.terminal_token_id:
            return output


@dataclass
class ModelContext:
    model: torch.nn.Module
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ]
    k: int
    terminal_token_id: int
    max_gen_horizon: int
    num_beams: int = NUM_BEAMS

    def __post_init__(self):
        self.cache = OutputTrie(self.terminal_token_id)
        self.generations = 0

    def _generate(self, ids: List[int], next_token_only: bool = False):
        input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)
        kwargs = (
            {"max_new_tokens": 1}
            if next_token_only
            else {"max_length": self.max_gen_horizon}
        )
        output = self.model.generate(
            input_ids,
            num_beams=self.num_beams,
            early_stopping=self.num_beams > 1,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            do_sample=False,
            **kwargs,
        )  # type: ignore
        (sequence,) = output.sequences
        sequence = sequence.squeeze(0).tolist()
        scores = [scores.squeeze(0) for scores in output.scores]
        self.generations += 1
        return {"sequence": sequence, "scores": scores}

    def generate(self, ids: List[int], next_token_only: bool = False):
        output = self.cache.search(ids, next_token_only)
        if output:
            return output
        output = self._generate(ids, next_token_only)
        self.cache.insert(
            output["sequence"],
            output["scores"],
        )
        return output


class Node:
    def __init__(
        self,
        state: List[int],
        action: Optional[int],
        prob: Optional[float],
        parent: Optional["Node"],
        model_context: ModelContext,
    ) -> None:
        self.state = state
        self.action = action
        self.prob = prob
        self.parent = parent
        self.ctx = model_context
        self.visits = 1
        self.observed_rewards = []
        self._children = []

    @property
    def display_action(self):
        return "root" if self.action is None else self.action

    @property
    def is_leaf_node(self) -> bool:
        """
        If we have not yet expanded this node, it is a leaf node.
        """
        return not bool(self._children)

    @property
    def is_terminal(self):
        return self.state[-1] == self.ctx.terminal_token_id

    @property
    def children(self):
        if not self._children and not self.is_terminal:
            self._children = self._generate_children()
        return self._children

    @property
    def value(self):
        if not self.observed_rewards:
            return 0
        return max(self.observed_rewards)

    def _generate_children(self):
        output = self.ctx.generate(self.state, next_token_only=True)
        (scores,) = output["scores"]
        top_k_scores, top_k_ids = torch.topk(scores, K)
        top_k_scores = torch.softmax(top_k_scores, dim=-1)
        children = []
        for id, score in zip(top_k_ids, top_k_scores):
            action = id.item()
            state = self.state + [action]
            prob = score.item()
            children.append(
                Node(
                    state=state,
                    action=action,
                    prob=prob,
                    parent=self,
                    model_context=self.ctx,
                )
            )
        return children


class Policy:
    def __init__(self, base: float = UCB_BASE, constant: float = UCB_CONSTANT):
        self.base = base
        self.constant = constant

    def compute_upper_confidence_bound(self, node: Node) -> float:
        """
        Compute the upper confidence bound of the node
        via the "Var P UCT" algorithm.
        """
        if not node.parent:
            raise Exception("Node has no parent; cannot compute UCB.")
        param = (
            np.log((node.parent.visits + self.base + 1) / self.base)
            + self.constant  # noqa: E501
        )
        return node.value + param * node.prob * np.sqrt(
            np.log(node.parent.visits)
        ) / (  # noqa: E501
            1 + len(node.observed_rewards)
        )

    def __call__(self, node: Node) -> float:
        return self.compute_upper_confidence_bound(node)


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
    root: Optional[Node],
    node: Node,
    rollout_index: Optional[int],
    token: Optional[str],
    elapsed: Optional[float],
):
    print(
        f"Step: {('Prediction' if elapsed is not None else 'Selection'):<10} |",  # noqa: E501
        f"Action #: {num_actions:<2} |",
        f"State 'Tip': {root.state[-1] if root else 'N/A':<6} |",
        f"Rollout #: {rollout_index if rollout_index is not None else 'N/A':<4} |",  # noqa: E501
        f"Action: {node.display_action if node else 'N/A':<6} |",
        f"Token: {repr(token) if token is not None else 'N/A':<6} |",
        f"Elapsed: {(str(np.round(elapsed, 3)) + 's' if elapsed is not None else 'N/A'):<7} |",  # noqa: E501
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode", default=False
    )
    parser.add_argument(
        "--K", type=int, help="Number of expanded children", default=K
    )  # noqa: E501
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    args = parser.parse_args()
    # Setup
    print("Loading model, tokenizer, etc...")
    device = (
        torch.device("cuda")
        if torch.cuda.is_available() and not NO_CUDA
        else torch.device("cpu")
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, pad_token_id=tokenizer.eos_token_id
    )
    model.to(device)
    (terminal_token_id,) = tokenizer.encode(TERMINAL_TOKEN)
    model_context = ModelContext(
        model,
        tokenizer,
        k=args.K,
        terminal_token_id=terminal_token_id,
        max_gen_horizon=MAX_GEN_HORIZON,
    )  # noqa: E501
    policy = Policy()

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Load problem

    # Run MCTS
    print("Running MCTS...")
    problem = Problem(TEST_PROBLEMS_DIR, TEST_PROBLEM_INDEX)
    state = tokenizer.encode(problem.prompt)
    node = root = Node(
        state=state,
        action=None,
        prob=None,
        parent=None,
        model_context=model_context,
    )
    num_actions = 0
    total_elapsed = 0
    rewards_cache = {}
    result = list(state)
    while True:
        start = time()
        # Perform rollouts
        for i in range(1, args.num_rollouts + 1):
            # Start at root
            node = root
            # Selection (select a leaf node)
            while True:
                if node.is_leaf_node:
                    if args.debug:
                        log_info(num_actions, root, node, i, None, None)
                    break
                node = max(node.children, key=policy)
            if not node.state[-1] == terminal_token_id:
                # Expansion (expand children, select one to rollout)
                # NB: If scores are the same, first node will always be selected.  # noqa: E501
                node = max(node.children, key=policy)
                # Simulate (simulate rollout)
                output = model_context.generate(node.state)
                text = model_context.tokenizer.decode(output["sequence"])
                code = extract_code(text)
                # Compute reward
                key = (code, problem)
                if key not in rewards_cache:
                    rewards_cache[key] = compute_reward(code, problem)
                reward = rewards_cache[key]
                # Backpropagation (update node statistics)
                while node:
                    node.visits += 1
                    node.observed_rewards.append(reward)
                    node = node.parent
        # Take action, reset root
        node = root = max(root.children, key=lambda node: node.value)
        # Log action
        elapsed = time() - start
        log_info(
            num_actions,
            None,
            node,
            None,
            tokenizer.decode(node.action),
            elapsed,
        )
        result.append(node.action)
        num_actions += 1
        total_elapsed += elapsed
        # Check if we're done
        if node.state[-1] == terminal_token_id:
            break
    code = extract_code(tokenizer.decode(result))
    reward = compute_reward(code, problem)
    print(f"\n>>> Result:\n\n{code}")
    print(
        f"\n>>> Reward: {reward} | Elapsed: {total_elapsed:.3f}s | Generations: {model_context.generations}"  # noqa: E501
    )
