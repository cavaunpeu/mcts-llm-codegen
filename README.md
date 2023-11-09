# MCTS with LLMs for Code Generation, via Modal

This repository contains a reimplementation of ["Planning with Large Language Models for Code Generation"](https://arxiv.org/abs/2303.05510) by Zhang et al. In addition, it contains a novel, trie-like caching structure for generated LLM sequences. Finally, it performs inference remotely, via [Modal](https://modal.com/)!

## How to run this code

1. First, clone this repository and submodules via: `git clone --recurse-submodules git@github.com:cavaunpeu/mcts-llm-codegen.git`.`
2. Ensure that you have [Docker](https://docs.docker.com/desktop/install/mac-install/) installed.
2. Follow Modal's simple setup instructions: `pip install modal && modal setup`. Once finished, you will have Modal credentials in `~/.modal.toml`. Modal offers you $30 of free compute credits per month, more than sufficient to run this code (on a remote GPU).
3. **Finally, run: `chmod u+x run/* && run/app.sh`.** This will generate code for [APPS](https://huggingface.co/datasets/codeparrot/apps) dataset test problem 4136, by default. Currently, we only offer problems `0001, 0002, 0003, 0004, 0005, 4136`.

## What do you mean by code generation?

The prompt for test problem 4136 is given below. The goal of code generation is to complete this prompt in a way that passes the associated test cases. A successful output might look like . "MCTS with LLMs," detailed by the authors above, is one way to do this.

```
QUESTION:
A + B is often used as an example of the easiest problem possible to show some contest platform. However, some scientists have observed
that sometimes this problem is not so easy to get accepted. Want to try?


-----Input-----

The input contains two integers a and b (0 ≤ a, b ≤ 10^3), separated by a single space.


-----Output-----

Output the sum of the given integers.


-----Examples-----
Input
5 14

Output
19

Input
381 492

Output
873
Use Standard Input format
ANSWER:
```