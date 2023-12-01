from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from app.util import get_wandb_runs
from scripts.const import GPU_TO_PRICE_PER_SECOND, GPU_TO_TFLOPS_PER_SECOND


def retrieve_and_preprocess_data():
    runs = get_wandb_runs("mcts-v1")
    runs_data = [{**run.config, **run.summary._json_dict} for run in runs]
    runs_df = pd.DataFrame(runs_data).rename(columns={"elapsed_ms": "elapsed_seconds"})
    runs_df = runs_df.assign(
        tflops_next_token_gen=lambda x: x["next_token_gen_time"]
        * x["device_name"].map(GPU_TO_TFLOPS_PER_SECOND),
        tflops_seq_gen=lambda x: x["seq_gen_time"]
        * x["device_name"].map(GPU_TO_TFLOPS_PER_SECOND),
        gen_time_perc_of_total=lambda x: (x["seq_gen_time"] + x["next_token_gen_time"])
        / x["elapsed_seconds"],
        model_size=lambda x: x["model_path"].str.extract(r"(\d+\.\d+)B").astype(float),
    )
    return runs_df


def plot_generation_time_distribution(runs_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=runs_df, x="gen_time_perc_of_total", bins=30, color="skyblue", kde=True
    )
    plt.xlabel("Generation Time Percentage of Total")
    plt.ylabel("Frequency")
    plt.title("Distribution of Generation Time as Percentage of Total Compute Time")
    sns.set_style("whitegrid")
    sns.despine(trim=True)
    plt.savefig("images/gen_time_distribution.png")


def plot_reward_tflops(agg_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Determine the y-axis limits for rewards (Train and Test)
    max_reward = max(agg_df["train_reward"].max(), agg_df["test_reward"].max())

    # First plot (Train Reward)
    sns.barplot(
        ax=axes[0, 0], data=agg_df, x="num_rollouts", y="train_reward", hue="model_size"
    )
    axes[0, 0].set_title("Mean Train Reward by Number of Rollouts and Model Size")
    axes[0, 0].set_xlabel("Number of Rollouts")
    axes[0, 0].set_ylabel("Train Reward")
    axes[0, 0].set_ylim(0, max_reward * 1.1)
    axes[0, 0].legend(
        title="Model Size (B)", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Second plot (Test Reward)
    sns.barplot(
        ax=axes[0, 1], data=agg_df, x="num_rollouts", y="test_reward", hue="model_size"
    )
    axes[0, 1].set_title("Mean Test Reward by Number of Rollouts and Model Size")
    axes[0, 1].set_xlabel("Number of Rollouts")
    axes[0, 1].set_ylabel("Test Reward")
    axes[0, 1].set_ylim(0, max_reward * 1.1)
    axes[0, 1].legend(
        title="Model Size (B)", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Determine the y-axis limits for generation TFLOPS (Sequence and Next Token)
    max_tflops = max(
        agg_df["tflops_next_token_gen"].max(), agg_df["tflops_seq_gen"].max()
    )

    # Third plot (Sequence Generation TFLOPS)
    sns.barplot(
        ax=axes[1, 0],
        data=agg_df,
        x="num_rollouts",
        y="tflops_seq_gen",
        hue="model_size",
    )
    axes[1, 0].set_title(
        "Mean Sequence Generation TFLOPS per Problem by Number of Rollouts and Model Size"
    )
    axes[1, 0].set_xlabel("Number of Rollouts")
    axes[1, 0].set_ylabel("Mean Sequence Generation TFLOPS per Problem")
    axes[1, 0].set_ylim(0, max_tflops * 1.1)
    axes[1, 0].legend(
        title="Model Size (B)", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    # Fourth plot (Next Token TFLOPS)
    sns.barplot(
        ax=axes[1, 1],
        data=agg_df,
        x="num_rollouts",
        y="tflops_next_token_gen",
        hue="model_size",
    )
    axes[1, 1].set_title(
        "Mean Next Token Generation TFLOPS per Problem by Number of Rollouts and Model Size"
    )
    axes[1, 1].set_xlabel("Number of Rollouts")
    axes[1, 1].set_ylabel("Mean Next Token Generation TFLOPS per Problem")
    axes[1, 1].set_ylim(0, max_tflops * 1.1)
    axes[1, 1].legend(
        title="Model Size (B)", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig("images/reward_tflops_plots.png")


def fit_models(agg_df):
    X = agg_df[["num_rollouts", "model_size"]]
    y_seq_gen = agg_df["tflops_seq_gen"]
    model_seq_gen = LinearRegression().fit(X, y_seq_gen)
    y_next_token_gen = agg_df["tflops_next_token_gen"]
    model_next_token_gen = LinearRegression().fit(X, y_next_token_gen)
    y_test_reward = agg_df["test_reward"]
    model_test_reward = LinearRegression().fit(X, y_test_reward)
    return model_seq_gen, model_next_token_gen, model_test_reward


def plot_model_fits(agg_df, model_seq_gen, model_next_token_gen, model_test_reward):
    X = agg_df[["num_rollouts", "model_size"]]
    _, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot for Test Reward Model
    r2_test_reward = r2_score(agg_df["test_reward"], model_test_reward.predict(X))
    sns.scatterplot(
        ax=axes[0],
        x="num_rollouts",
        y="test_reward",
        data=agg_df,
        hue="model_size",
        style="model_size",
        s=100,
    )
    sns.lineplot(
        ax=axes[0],
        x="num_rollouts",
        y=model_test_reward.predict(X),
        data=agg_df,
        color="red",
        label="Fitted Line",
    )
    axes[0].set_title("Test Reward Model Fit")
    axes[0].set_xlabel("Number of Rollouts")
    axes[0].set_ylabel("Test Reward")
    axes[0].legend(title="Model Size (B)", loc="best")
    axes[0].text(
        0.25,
        0.05,
        f"R-squared: {r2_test_reward:.2f}",
        transform=axes[0].transAxes,
        horizontalalignment="right",
    )  # Annotation
    axes[0].grid(True)

    # Plot for Mean Sequence Generation TFLOPS Model
    r2_seq_gen = r2_score(agg_df["tflops_seq_gen"], model_seq_gen.predict(X))
    sns.scatterplot(
        ax=axes[1],
        x="num_rollouts",
        y="tflops_seq_gen",
        data=agg_df,
        hue="model_size",
        style="model_size",
        s=100,
    )
    sns.lineplot(
        ax=axes[1],
        x="num_rollouts",
        y=model_seq_gen.predict(X),
        data=agg_df,
        color="blue",
        label="Fitted Line",
    )
    axes[1].set_title("Mean Sequence Generation TFLOPS Model Fit")
    axes[1].set_xlabel("Number of Rollouts")
    axes[1].set_ylabel("Mean Sequence Generation TFLOPS")
    axes[1].legend(title="Model Size (B)", loc="best")
    axes[1].text(
        0.25,
        0.05,
        f"R-squared: {r2_seq_gen:.2f}",
        transform=axes[1].transAxes,
        horizontalalignment="right",
    )  # Annotation
    axes[1].grid(True)

    # Plot for Mean Next Token Generation TFLOPS Model
    r2_next_token_gen = r2_score(
        agg_df["tflops_next_token_gen"], model_next_token_gen.predict(X)
    )
    sns.scatterplot(
        ax=axes[2],
        x="num_rollouts",
        y="tflops_next_token_gen",
        data=agg_df,
        hue="model_size",
        style="model_size",
        s=100,
    )
    sns.lineplot(
        ax=axes[2],
        x="num_rollouts",
        y=model_next_token_gen.predict(X),
        data=agg_df,
        color="green",
        label="Fitted Line",
    )
    axes[2].set_title("Mean Next Token Generation TFLOPS Model Fit")
    axes[2].set_xlabel("Number of Rollouts")
    axes[2].set_ylabel("Mean Next Token Generation TFLOPS")
    axes[2].legend(title="Model Size (B)", loc="best")
    axes[2].text(
        0.25,
        0.05,
        f"R-squared: {r2_next_token_gen:.2f}",
        transform=axes[2].transAxes,
        horizontalalignment="right",
    )  # Annotation
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("images/model_fit_plots.png")


def predict_test_reward_and_tflops(
    num_rollouts, model_size, model_seq_gen, model_next_token_gen, model_test_reward
):
    (predicted_seq_tflops,) = model_seq_gen.predict(
        pd.DataFrame(
            [[num_rollouts, model_size]], columns=["num_rollouts", "model_size"]
        )
    )
    (predicted_next_token_tflops,) = model_next_token_gen.predict(
        pd.DataFrame(
            [[num_rollouts, model_size]], columns=["num_rollouts", "model_size"]
        )
    )
    predicted_tflops = predicted_seq_tflops + predicted_next_token_tflops
    (predicted_test_reward,) = model_test_reward.predict(
        pd.DataFrame(
            [[num_rollouts, model_size]], columns=["num_rollouts", "model_size"]
        )
    )
    return predicted_test_reward, predicted_tflops


def plot_rewards_heatmap(
    num_rollouts_range,
    model_sizes,
    model_seq_gen,
    model_next_token_gen,
    model_test_reward,
):
    # Initialize matrices to store the predicted rewards and TFLOPS
    reward_matrix = np.zeros((len(num_rollouts_range), len(model_sizes)))
    tflops_matrix = np.zeros((len(num_rollouts_range), len(model_sizes)))

    # Iterate over combinations and fill the matrices
    for i, num_rollouts in enumerate(num_rollouts_range):
        for j, model_size in enumerate(model_sizes):
            reward_matrix[i, j], tflops_matrix[i, j] = predict_test_reward_and_tflops(
                num_rollouts,
                model_size,
                model_seq_gen,
                model_next_token_gen,
                model_test_reward,
            )

    # Create a DataFrames
    reward_df = pd.DataFrame(
        reward_matrix, index=num_rollouts_range, columns=np.round(model_sizes, 2)
    )
    tflops_df = pd.DataFrame(
        tflops_matrix, index=num_rollouts_range, columns=np.round(model_sizes, 2)
    )
    reward_df.index.name = "num_rollouts"
    tflops_df.index.name = "num_rollouts"

    # Plot the heatmap for rewards
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        reward_df,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        cbar_kws={"label": "Predicted Reward"},
    )
    plt.title("Heatmap of Predicted Test Reward")
    plt.xlabel("Model Size")
    plt.ylabel("Number of Rollouts")
    plt.savefig("images/rewards_heatmap.png")

    # Return DataFrames
    return reward_df, tflops_df


def compose_cost_dataframe(tflops_df, reward_df, runs_df):
    tflops_df_melted = tflops_df.reset_index().melt(
        id_vars="num_rollouts", var_name="model_size", value_name="predicted_tflops"
    )
    reward_df_melted = reward_df.reset_index().melt(
        id_vars="num_rollouts",
        var_name="model_size",
        value_name="predicted_test_reward",
    )
    pred_df = tflops_df_melted.merge(
        reward_df_melted, on=["num_rollouts", "model_size"]
    )

    # Compute costs in time and $
    num_problems = len(set(runs_df["problem_index"]))
    data = []

    for _, row in pred_df.iterrows():
        for gpu, tflops_per_second in GPU_TO_TFLOPS_PER_SECOND.items():
            d = row.to_dict()
            d["predicted_time"] = (
                num_problems * d["predicted_tflops"] / tflops_per_second
            )
            d["predicted_dollar_cost"] = (
                d["predicted_time"] * GPU_TO_PRICE_PER_SECOND[gpu]
            )
            d["predicted_time_minutes"] = d["predicted_time"] / 60
            d["gpu"] = gpu
            data.append(d)

    return pd.DataFrame(data)


def plot_budget_vs_optimal_setting_heatmap(costs_df):
    # Define the grid range for dollar cost and time
    dollar_cost_min, dollar_cost_max = (
        costs_df["predicted_dollar_cost"].min(),
        costs_df["predicted_dollar_cost"].max(),
    )
    time_minutes_min, time_minutes_max = (
        costs_df["predicted_time_minutes"].min(),
        costs_df["predicted_time_minutes"].max(),
    )

    # Define the number of points in the grid for both dimensions
    num_points = 10  # for example, you can increase this for a finer grid

    # Create the grid
    dollar_cost_grid = np.linspace(dollar_cost_min, dollar_cost_max, num_points)
    time_minutes_grid = np.linspace(time_minutes_min, time_minutes_max, num_points)

    # Initialize the list to store the best configurations
    best_configs = []

    # Iterate over the grid
    for dollar_cost in dollar_cost_grid:
        for time_minutes in time_minutes_grid:
            # Filter rows that meet the current constraints
            filtered_df = costs_df[
                (costs_df["predicted_dollar_cost"] <= dollar_cost)
                & (costs_df["predicted_time_minutes"] <= time_minutes)
            ]

            # If there are any rows that meet the condition, select the one with the highest predicted_test_reward  # noqa: E501
            if not filtered_df.empty:
                best_row = filtered_df.loc[
                    filtered_df["predicted_test_reward"].idxmax()
                ]
                best_configs.append(
                    {
                        "dollar_cost": dollar_cost,
                        "time_minutes": time_minutes,
                        "predicted_test_reward": best_row["predicted_test_reward"],
                        "gpu": best_row["gpu"],
                        "num_rollouts": best_row["num_rollouts"],
                        "model_size": best_row["model_size"],
                    }
                )

    # Convert the best configurations to a DataFrame
    best_configs_df = pd.DataFrame(best_configs)

    # Pivot 'best_configs_df'
    heatmap_test_reward = best_configs_df.pivot(
        index="dollar_cost", columns="time_minutes", values="predicted_test_reward"
    )
    heatmap_gpu = best_configs_df.pivot(
        index="dollar_cost", columns="time_minutes", values="gpu"
    )
    heatmap_num_rollouts = best_configs_df.pivot(
        index="dollar_cost", columns="time_minutes", values="num_rollouts"
    )
    heatmap_model_size = best_configs_df.pivot(
        index="dollar_cost", columns="time_minutes", values="model_size"
    )

    plt.figure(figsize=(18, 8))
    cmap = sns.light_palette("orange")
    ax = sns.heatmap(
        heatmap_test_reward,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        cbar_kws={"label": "Predicted Test Reward"},
    )

    # Annotate each cell with the combined information
    for i, row in enumerate(heatmap_test_reward.values):
        for j, val in enumerate(row):
            gpu = heatmap_gpu.iloc[i, j]
            num_rollouts = heatmap_num_rollouts.iloc[i, j]
            model_size = heatmap_model_size.iloc[i, j]
            reward_val = val if pd.notnull(val) else 0
            text = f"{reward_val:.4f}\n{gpu}, ({num_rollouts}, {model_size})"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9)

    plt.title(
        "Heatmap of Predicted Test Reward\nAnnotated with: GPU, (num_rollouts, model_size)"
    )
    plt.xlabel("Time Budget (Minutes)")
    plt.ylabel("Cost Budget ($)")

    # Round the tick labels to 3 significant digits
    ax.set_xticklabels(
        ["{:.3g}".format(float(t.get_text())) for t in ax.get_xticklabels()]
    )
    ax.set_yticklabels(
        ["{:.3g}".format(float(t.get_text())) for t in ax.get_yticklabels()]
    )

    # Save the figure
    plt.savefig("images/final_heatmap.png")
