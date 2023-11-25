from const import MODEL_BASE_PATH


experiments = {
    "mcts": {
        "k": [3],
        "num_rollouts": [2, 4, 6],
        "model_path": [f"{MODEL_BASE_PATH}/1.5B", f"{MODEL_BASE_PATH}/2.7B"],
    },
}
