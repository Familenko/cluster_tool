import numpy as np
import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


class AgglomerativeCluster:
    def agglo_distance(
            self,
            min_d: int = 1,
            max_d: int = 5,
            range_d: int = 10
    ):
        # DESCRIPTION:

        #     Build diagram with distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_d - minimum level of distance
        #     max_d - maximum level of distance
        #     range_d - amount of tested distances

        distance_threshold_list = list(np.linspace(min_d, max_d, range_d))

        n_cluster_dis = []
        distance_threshold = []
        silhouette_dis = []

        for distance_threshold_n in tqdm(
                distance_threshold_list,
                desc="Checking distance"
        ):

            cluster = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold_n
            )
            cluster_labels = cluster.fit_predict(self.X)

            distance_threshold.append(
                distance_threshold_n
            )
            n_cluster_dis.append(
                len(np.unique(cluster_labels))
            )
            silhouette_dis.append(
                round(silhouette_score(self.X, cluster_labels), 3),
            )

        plt.figure(figsize=(12, 5), dpi=200)
        sns.scatterplot(
            x=n_cluster_dis,
            y=distance_threshold,
            hue=silhouette_dis,
            s=100
        )

        plt.title("Distance plot")
        plt.xlabel("Number of clusters")
        plt.ylabel("Distance threshold")
        plt.xticks(range(min(n_cluster_dis), max(n_cluster_dis) + 1))
        plt.yticks(np.arange(0, max(distance_threshold) + 1, step=1))
        plt.legend(title="Silhouette", loc="upper right")

        plt.show()

        df = pd.DataFrame(
            {
                "number_of_clusters": n_cluster_dis,
                "distance_threshold": distance_threshold,
                "silhouette": silhouette_dis,
            }
        )

        return df.transpose()

    def agglo_knife(
            self,
            min_k: int,
            max_k: int,
            step=1,
            knife: bool = True,

    ):
        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakages

        # ARGUMENTS:

        #     min_k - minimum amount of cluster
        #     max_k - maximum amount of cluster

        range_n_clusters = list(range(min_k, max_k, step))

        n_clus = []
        silhouette = []

        for n_clusters in tqdm(
                range_n_clusters,
                desc="Checking knifes"
        ):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            cluster = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = cluster.fit_predict(self.X)

            n_clus.append(n_clusters)
            silhouette.append(round(
                silhouette_score(self.X, cluster_labels), 3)
            )

            print(
                "n_clusters =",
                n_clusters,
                "average silhouette_score =",
                round(silhouette_score(self.X, cluster_labels), 3))

            if knife:

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
                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,)

                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10

                ax1.set_title("Silhouette plot for n_clusters = %d" % n_clusters)
                ax1.set_xlabel("Silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                ax1.axvline(x=silhouette_score(self.X, cluster_labels), color="red", linestyle="--")
                ax1.set_yticks([])
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                plt.show()

        df = pd.DataFrame({
            'silhouette': silhouette},
            index=range(min_k, max_k, step))

        return df.transpose()
