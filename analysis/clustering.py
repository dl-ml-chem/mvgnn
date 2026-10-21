# 用于 latent embedding 聚类分析 + t-SNE 可视化。

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


def run_clustering(X, fig_dir="figures"):

    os.makedirs(fig_dir, exist_ok=True)

    km = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)

    print("\nClustering Results")
    print("Silhouette:", round(sil, 4))
    print("DB Index :", round(db, 4))

    tsne = TSNE(
        n_components=2,
        random_state=42
    )

    z = tsne.fit_transform(X)

    plt.figure(figsize=(7, 5))

    plt.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        s=80
    )

    plt.title("Latent Space Clustering")
    plt.tight_layout()

    plt.savefig(
        os.path.join(fig_dir, "cluster_tsne.png"),
        dpi=300
    )