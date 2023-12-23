import numpy as np
import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

import matplotlib.pyplot as plt
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
