from dataclasses import dataclass
import re
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
class ModelContext:
    model: torch.nn.Module
    tokenizer: Union[
        transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
    ]
    k: int
    max_gen_horizon: int
    num_beams: int = NUM_BEAMS

    def __post_init__(self):
        self.cache = {}

    def generate(self, ids: List[int], next_token_only: bool = False):
        key = (tuple(ids), next_token_only)
        if key not in self.cache:
            input_ids = torch.LongTensor(ids).unsqueeze(0).to(device)
            kwargs = (
                {"max_new_tokens": 1}
                if next_token_only
                else {"max_length": self.max_gen_horizon}
            )
            self.cache[key] = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                early_stopping=self.num_beams > 1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
                do_sample=self.k > 1,
                **kwargs,
            )  # type: ignore
        return self.cache[key]


class Node:
    def __init__(
        self,
        state: List[int],
        action: Optional[int],
        prob: Optional[float],
        parent: Optional["Node"],
        model_context: ModelContext,
        terminal_token_id: int,
    ) -> None:
        self.state = state
        self.action = action
        self.prob = prob
        self.parent = parent
        self.ctx = model_context
        self.terminal_token_id = terminal_token_id
        self.visits = 1
        self.observed_rewards = []
        self._children = []

    @property
    def is_leaf_node(self) -> bool:
        """
        If we have not yet expanded this node, it is a leaf node.
        """
        return not bool(self._children)

    @property
    def is_terminal(self):
        return self.state[-1] == self.terminal_token_id

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
        (scores,) = output.scores
        scores = scores.squeeze(0)
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
                    terminal_token_id=self.terminal_token_id,
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


if __name__ == "__main__":
    # Setup model
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

    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Load problem
    problem = Problem(TEST_PROBLEMS_DIR, TEST_PROBLEM_INDEX)
    state = tokenizer.encode(problem.prompt)

    # Run MCTS
    model_context = ModelContext(model, tokenizer, max_gen_horizon=MAX_GEN_HORIZON, k=K)
    policy = Policy()
    node = root = Node(
        state=state,
        action=None,
        prob=None,
        parent=None,
        model_context=model_context,
        terminal_token_id=terminal_token_id,
    )
    plan = True
    num_actions = 0
    while plan:
        # Perform rollouts
        for i in range(1, NUM_ROLLOUTS + 1):
            # Start at root
            node = root
            level = 0
            # Selection (select a leaf node)
            while True:
                if node.is_leaf_node:
                    action = "Root" if node == root else node.action
                    print(
                        f"Selected | Action #: {num_actions:<2} | Action: {action:<5} | Level: {level:<2} | Rollout #: {i}"  # noqa: E501
                    )
                    break
                node = max(node.children, key=policy)
                level += 1
            if node.state[-1] == terminal_token_id:
                plan = False
                break
            # Expansion (expand children, select one to rollout)
            # NB: If scores are the same, first node will always be selected.
            node = max(node.children, key=policy)
            # Simulate (simulate rollout)
            (ids,) = model_context.generate(node.state).sequences.tolist()
            text = model_context.tokenizer.decode(ids)
            code = extract_code(text)
            reward = compute_reward(code, problem)
            # Backpropagation (update node statistics)
            while node:
                node.visits += 1
                node.observed_rewards.append(reward)
                node = node.parent
        num_actions += 1