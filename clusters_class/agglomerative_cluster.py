import numpy as np
import pandas as pd

import tqdm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

from utilities import knife_show


class AgglomerativeCluster:
    def agglo_distance(
            self,
            min_distance_threshold: int = 1,
            max_distance_threshold: int = 5,
            range_distance_threshold: int = 10,
            **kwargs
    ):

        # DESCRIPTION:

        #     Build diagram with distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_d - minimum level of distance
        #     max_d - maximum level of distance
        #     range_d - amount of tested distances

        distance_threshold_list = list(
            np.linspace(min_distance_threshold,
                        max_distance_threshold,
                        range_distance_threshold)
        )

        n_cluster_dis = []
        distance_threshold = []
        silhouette_dis = []

        for distance_threshold_n in tqdm.tqdm(distance_threshold_list):
            cluster = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold_n,
                **kwargs
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
            min_cluster: int = 2,
            max_cluster: int = 10,
            step_cluster=1,
            knife: bool = True,
            **kwargs
    ):
        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakages

        # ARGUMENTS:

        #     min_k - minimum amount of cluster
        #     max_k - maximum amount of cluster

        range_n_clusters = list(range(min_cluster, max_cluster, step_cluster))

        n_clus = []
        silhouette = []

        for n_clusters in tqdm.tqdm(range_n_clusters):

            try:
                cluster = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    **kwargs
                )
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

                    knife_show(
                        self,
                        cluster_labels,
                        n_clusters,
                    )

            except ValueError:
                continue

        df = pd.DataFrame({
            'silhouette': silhouette},
            index=range(min_cluster, max_cluster, step_cluster))

        return df.transpose()
