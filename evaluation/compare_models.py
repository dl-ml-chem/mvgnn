import os
import pandas as pd
import matplotlib.pyplot as plt


def save_results(results, fig_dir="figures"):

    os.makedirs(fig_dir, exist_ok=True)

    df = pd.DataFrame(results)

    df = df.sort_values("RMSE")

    df.to_csv(
        os.path.join(fig_dir, "benchmark.csv"),
        index=False
    )

    print("\nBenchmark Results")
    print(df)

    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df["RMSE"])

    plt.xticks(rotation=30, ha="right")

    plt.title("Model Comparison (RMSE)")
    plt.tight_layout()

    plt.savefig(
        os.path.join(fig_dir, "rmse_compare.png"),
        dpi=300
    )