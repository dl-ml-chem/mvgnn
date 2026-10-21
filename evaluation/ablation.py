import os
import pandas as pd
import matplotlib.pyplot as plt


def save_ablation(rows, fig_dir="figures"):

    os.makedirs(fig_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    df.to_csv(
        os.path.join(fig_dir, "ablation.csv"),
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df["RMSE"])

    plt.xticks(rotation=25, ha="right")

    plt.title("Ablation Study")

    plt.tight_layout()

    plt.savefig(
        os.path.join(fig_dir, "ablation_rmse.png"),
        dpi=300
    )