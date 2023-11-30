import os
import numpy as np
import matplotlib

from scripts.util import (
    compose_cost_dataframe,
    fit_models,
    plot_budget_vs_optimal_setting_heatmap,
    plot_generation_time_distribution,
    plot_model_fits,
    plot_reward_tflops,
    plot_rewards_heatmap,
    retrieve_and_preprocess_data,
)

matplotlib.use("Agg")


MAX_TFLOPS_BUDGET = 3e3
NUM_ROLLOUTS_RANGE = range(2, 6)
MODEL_SIZES = np.linspace(1.5, 3.0, 16)
GROUP_COLS = ["k", "num_rollouts", "model_size"]


def main():
    runs_df = retrieve_and_preprocess_data()
    os.makedirs("images", exist_ok=True)

    # Aggregate runs data
    aggregations = {
        "tflops_next_token_gen": "mean",
        "tflops_seq_gen": "mean",
        "next_token_gen_time": "mean",
        "seq_gen_time": "mean",
        "train_reward": "mean",
        "test_reward": "mean",
    }
    agg_df = runs_df.groupby(GROUP_COLS).agg(aggregations).reset_index()
    agg_df["model_size"] = agg_df["model_size"].astype("category")

    # Plot generation time distribution
    plot_generation_time_distribution(runs_df)

    # Plot reward and TFLOPS
    plot_reward_tflops(agg_df)

    # Fit models and plot
    model_seq_gen, model_next_token_gen, model_test_reward = fit_models(agg_df)
    plot_model_fits(agg_df, model_seq_gen, model_next_token_gen, model_test_reward)

    # Plot rewards heatmap
    reward_df, tflops_df = plot_rewards_heatmap(
        NUM_ROLLOUTS_RANGE,
        MODEL_SIZES,
        model_seq_gen,
        model_next_token_gen,
        model_test_reward,
    )

    # Compose cost dataframe
    costs_df = compose_cost_dataframe(tflops_df, reward_df, runs_df)

    # Plot budget vs optimal setting heatmap
    plot_budget_vs_optimal_setting_heatmap(costs_df)


if __name__ == "__main__":
    main()
