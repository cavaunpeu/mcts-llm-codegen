from const import MODEL_BASE_PATH


experiments = {
    "mcts": {
        "k": [3],
        "num_rollouts": [5],
        "model_path": [f"{MODEL_BASE_PATH}/1.5B", f"{MODEL_BASE_PATH}/2.7B"],
    },
    "test-remote-logging": {
        "k": [3],
        "num_rollouts": [2],
        "model_path": [f"{MODEL_BASE_PATH}/1.5B"],
    },
    "mcts-v1": {
        "k": [3],
        "num_rollouts": [2],
        "model_path": [f"{MODEL_BASE_PATH}/1.5B", f"{MODEL_BASE_PATH}/2.7B"],
    },
}
