# 模型解释模块。

import os
import torch
import matplotlib.pyplot as plt


def run_shap(student, X, fig_dir="figures"):

    try:
        import shap

        os.makedirs(fig_dir, exist_ok=True)

        def f(v):

            x = torch.tensor(
                v,
                dtype=torch.float
            )

            with torch.no_grad():
                p = student(x).numpy()

            return p

        explainer = shap.Explainer(f, X)

        sv = explainer(X)

        shap.plots.beeswarm(
            sv,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(fig_dir, "shap.png"),
            dpi=300
        )

        print("SHAP saved.")

    except:
        print("Please install shap.")