from collections import defaultdict
from dataclasses import dataclass
import os
from typing import List, Optional, Union

import numpy as np
import torch
import transformers

from const import (
    APPS_PROBLEMS_DIR,
    MAX_GEN_HORIZON,
    MODEL_NAME,
    MODEL_PATH,
    NO_CUDA,
    NUM_BEAMS,
    SEED,
    TERMINAL_TOKEN,
    UCB_BASE,
    UCB_CONSTANT,
)


class APPSProblem:
    base_path = APPS_PROBLEMS_DIR
    problem_indices = os.listdir(base_path)

    def __init__(self, idx: int):
        self.dir = f"{self.base_path}/{idx}"
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


class ModelContext:
    def __init__(
        self,
        model_path: str,
        model_name: str = MODEL_NAME,
        terminal_token: str = TERMINAL_TOKEN,
        max_gen_horizon: int = MAX_GEN_HORIZON,
        no_cuda: bool = NO_CUDA,
        num_beams: int = NUM_BEAMS,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.terminal_token = terminal_token
        self.max_gen_horizon = max_gen_horizon
        self.no_cuda = no_cuda
        self.num_beams = num_beams

    def initialize(self):
        # Initialize
        self.num_sequence_gens = 0
        self.num_next_token_gens = 0
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        (self.terminal_token_id,) = tokenizer.encode(self.terminal_token)
        self.tokenizer = tokenizer
        # Initialize cache
        self.cache = OutputTrie(self.terminal_token_id)
        # Set seed
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        # Setup device
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and not self.no_cuda
            else torch.device("cpu")
        )
        # Load model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, pad_token_id=self.tokenizer.eos_token_id
        )
        self.model.to(self.device)

    def _generate(self, ids: List[int], next_token_only: bool = False):
        input_ids = torch.LongTensor(ids).unsqueeze(0).to(self.device)
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
        )
        # Update counters
        if next_token_only:
            self.num_next_token_gens += 1
        else:
            self.num_sequence_gens += 1
        # Extract output
        (sequence,) = output.sequences
        sequence = sequence.squeeze(0).tolist()
        scores = [scores.squeeze(0).cpu() for scores in output.scores]
        return {"sequence": sequence, "scores": scores}

    def generate(
        self,
        ids: List[int],
        next_token_only: bool = False,
    ):
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
        action: Optional[Union[int, str]],
        prob: Optional[float],
        parent: Optional["Node"],
        model_context: ModelContext,  # type: ignore
        k: int,
    ) -> None:
        self.state = state
        self.action = action
        self.prob = prob
        self.parent = parent
        self.ctx = model_context
        self.k = k
        self.visits = 0
        self.selected = 0
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
        top_k_scores, top_k_ids = torch.topk(scores, self.k)
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
                    k=self.k,
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
            np.log(node.parent.visits) if node.parent.visits >= 1 else 0
        ) / (  # noqa: E501
            1 + len(node.observed_rewards)
        )

    def __call__(self, node: Node) -> float:
        return self.compute_upper_confidence_bound(node)
