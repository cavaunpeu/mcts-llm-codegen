import torch
import pytest

from type import OutputTrie


@pytest.fixture
def trie():
    terminal_token_id = 5
    return OutputTrie(terminal_token_id)


@pytest.fixture
def input_seq():
    return [1, 2, 3]


@pytest.fixture
def output_seq():
    return [4, 5]


@pytest.fixture
def output_scores(output_seq):
    return [torch.tensor(i) for i in output_seq]


def test_empty_trie_search(trie, input_seq):
    result = trie.search(input_seq)
    assert result is None


def test_insert_and_search(trie, input_seq, output_seq, output_scores):
    inp, out = input_seq, output_seq
    seq = inp + out
    trie.insert(seq, output_scores)
    for i in range(len(output_seq)):
        search_seq = inp + out[:i]
        num_tokens_to_gen = len(seq) - len(search_seq)
        result = trie.search(search_seq)
        assert result["sequence"] == seq
        assert result["scores"] == output_scores[-num_tokens_to_gen:]


def test_insert_and_search_modified_sequence(
    trie, input_seq, output_seq, output_scores
):
    seq = input_seq + output_seq
    trie.insert(seq, output_scores)
    input_seq[-1] = 100
    result = trie.search(input_seq)
    assert result is None


def test_next_token_only(trie, input_seq, output_seq, output_scores):
    inp, out = input_seq, output_seq
    seq = inp + out
    trie.insert(seq, output_scores)
    for i in range(len(output_seq)):
        search_seq = inp + out[:i]
        num_tokens_to_gen = len(seq) - len(search_seq)
        result = trie.search(search_seq, next_token_only=True)
        assert result["sequence"] == seq[: (len(search_seq) + 1)]
        assert (
            result["scores"]
            == output_scores[-num_tokens_to_gen : -num_tokens_to_gen + 1 or None]
        )
