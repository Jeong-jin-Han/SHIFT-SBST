import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# create plot directory
PLOT_DIR = Path("test4_plots")
PLOT_DIR.mkdir(exist_ok=True)


def load_algorithm_results(base_paths):
    """
    base_paths: dict
        key = algorithm name
        value = directory path containing seed folders

    Example:
        base_paths = {
            "hcc_biased": "benchmark_log_test4_hcc_biased_test",
            "ga_biased": "benchmark_log_test4_ga_biased_test",
            ...
        }
    """
    all_rows = []

    for algo, base_dir in base_paths.items():
        for seed_dir in sorted(Path(base_dir).glob("seed_*")):
            seed = int(str(seed_dir).split("_")[-1])

            for csv_file in seed_dir.rglob("*.csv"):
                df = pd.read_csv(csv_file)
                df["algorithm"] = algo
                df["seed"] = seed
                df["branch"] = (
                    df["function"].astype(str)
                    + ":"
                    + df["lineno"].astype(str)
                    + ":"
                    + df["outcome"].astype(str)
                )
                all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


# Seed variance boxplots
def plot_seed_variance(df, metrics=["best_fitness", "nfe", "convergence_speed"]):
    algorithms = sorted(df["algorithm"].unique())

    for algo in algorithms:
        df_algo = df[df["algorithm"] == algo]

        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="seed", y=metric, data=df_algo)
            # plt.title(f"Seed Variance for {algo} — {metric}")
            plt.tight_layout()
            # save in test4_analysis_plots folder
            plt.savefig(
                PLOT_DIR / f"{algo}_{metric}_seed_variance_boxplot.png", dpi=300
            )
            # save pdf
            plt.savefig(PLOT_DIR / f"{algo}_{metric}_seed_variance_boxplot.pdf")
            plt.close()


# Algorithm-wise boxplot (best_fitness)
def plot_algorithm_boxplot(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="algorithm", y="best_fitness", data=df)
    # plt.title("Algorithm Comparison — Best Fitness")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "algorithm_best_fitness_boxplot.png", dpi=300)
    plt.savefig(PLOT_DIR / "algorithm_best_fitness_boxplot.pdf")
    plt.close()


# Seed sensitivity heatmap (success rate)
def plot_seed_sensitivity_by_algorithm(df):
    # success를 평균하면 = success rate (0~1)
    pivot = df.pivot_table(
        index="algorithm", columns="seed", values="success", aggfunc="mean"
    )

    # Dynamic range normalization (makes colors more sensitive)
    vmin = pivot.values.min()
    vmax = pivot.values.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".2f", norm=norm)
    # plt.title("Seed Sensitivity Heatmap (Success Rate)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "seed_sensitivity_successrate.png", dpi=300)
    plt.savefig(PLOT_DIR / "seed_sensitivity_successrate.pdf")
    plt.close()


# Seed sensitivity heatmap (convergence speed)
def plot_seed_sensitivity_by_algorithm_convergence(df):
    pivot = df.pivot_table(
        index="algorithm", columns="seed", values="convergence_speed", aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="magma")
    # plt.title("Seed Sensitivity Heatmap (Convergence Speed)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "seed_sensitivity_convergence.png", dpi=300)
    plt.savefig(PLOT_DIR / "seed_sensitivity_convergence.pdf")
    plt.close()


# Branch difficulty heatmap (success rate)
def plot_branch_difficulty(df):
    pivot = (
        df.pivot_table(
            index="branch", columns="algorithm", values="success", aggfunc="mean"
        )
        * 100.0
    )  # convert to percent

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap="coolwarm", annot=False)
    # plt.title("Branch Difficulty Heatmap (Success Rate %)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "branch_difficulty_heatmap.png", dpi=300)
    plt.savefig(PLOT_DIR / "branch_difficulty_heatmap.pdf")
    plt.close()


# Algorithm Sensitivity Summary (single-table, one-glance view)
def compute_algorithm_sensitivity(df):
    metrics = ["best_fitness", "nfe", "convergence_speed"]
    algos = sorted(df["algorithm"].unique())

    rows = []

    for algo in algos:
        df_algo = df[df["algorithm"] == algo]

        row = {"algorithm": algo}

        for metric in metrics:
            mean = df_algo.groupby("seed")[metric].mean().mean()
            std = df_algo.groupby("seed")[metric].mean().std()

            cv = (std / mean) * 100 if mean != 0 else 0

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_cv"] = cv

        # 전체 sensitivity로 종합 score 생성
        row["sensitivity_score"] = (
            row["best_fitness_cv"] + row["nfe_cv"] + row["convergence_speed_cv"]
        ) / 3.0

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(PLOT_DIR / "algorithm_sensitivity_summary.csv", index=False)
    print("\n📊 Algorithm Sensitivity Summary:")
    print(summary_df)

    return summary_df


# Sensitivity Bar Chart
def plot_sensitivity_score(summary_df):
    plt.figure(figsize=(10, 5))
    sns.barplot(
        x="algorithm", y="sensitivity_score", data=summary_df, palette="viridis"
    )
    # plt.title("Overall Algorithm Seed Sensitivity Score (Lower = More Stable)")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "algorithm_seed_sensitivity_score.png", dpi=300)
    plt.savefig(PLOT_DIR / "algorithm_seed_sensitivity_score.pdf")
    plt.close()


if __name__ == "__main__":

    base_paths = {
        "hcc_biased": "benchmark_log_test4_hcc_biased_test",
        "hcc_random": "benchmark_log_test4_hcc_random_test",
        "ga_biased": "benchmark_log_test4_ga_biased_test",
        "ga_random": "benchmark_log_test4_ga_random_test",
        "hc_biased": "benchmark_log_test4_hc_biased_test",
        "hc_random": "benchmark_log_test4_hc_random_test",
    }

    print("📂 Loading CSV result files...")
    # get seed from 41-45
    # df = load_algorithm_results(base_paths)
    df = load_algorithm_results(base_paths)
    df = df[df["seed"].isin([41, 42, 43, 44, 45])]
    print(f"Loaded {len(df)} rows.")

    print("\n[A] Seed variance boxplots…")
    plot_seed_variance(df)

    print("\n[B] Algorithm-wise best_fitness comparison…")
    plot_algorithm_boxplot(df)

    print("\n[C] Seed sensitivity heatmap by success rate…")
    plot_seed_sensitivity_by_algorithm(df)
    print("\n[C] Seed sensitivity heatmap by convergence speed…")
    plot_seed_sensitivity_by_algorithm_convergence(df)

    print("\n[D] Branch difficulty heatmap…")
    plot_branch_difficulty(df)

    print("\n📈 Computing algorithm sensitivity summary…")
    summary_df = compute_algorithm_sensitivity(df)
    plot_sensitivity_score(summary_df)

    print("\n✨ All plots generated.")
