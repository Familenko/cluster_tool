import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples

import matplotlib.pyplot as plt
import seaborn as sns


def knife_show(
        self: object,
        cluster_labels: np.ndarray,
        n_clusters: int,
):
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(7, 4)

    sample_silhouette_values = silhouette_samples(
        self.X,
        cluster_labels
    )

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = (
            sample_silhouette_values[cluster_labels == i]
        )

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cluster_colors = sns.color_palette(
            "Set1",
            n_colors=n_clusters
        )
        color = cluster_colors[i % n_clusters]

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7, )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for n_clusters = {n_clusters}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(
        x=silhouette_score(self.X, cluster_labels),
        color="red",
        linestyle="--"
    )
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
