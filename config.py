from const import MODEL_PATH


experiments = {
    "mcts": {
        "param_ranges": {"K": [3], "num_rollouts": [2, 4, 6, 8]},
        "model_paths": [MODEL_PATH],
    }
}
