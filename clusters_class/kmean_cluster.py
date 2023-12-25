import pandas as pd

import tqdm

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

from utilities import knife_show


class KMeansCluster:
    def kmean_mount(
            self,
            min_cluster: int = 2,
            max_cluster: int = 10,
            **kwargs
    ):

        # DESCRIPTION:

        #     Build diagram with squared distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_n - minimum cluster
        #     max_n - maximum cluster

        range_n_clusters = list(range(min_cluster, max_cluster))

        n_clus_mount = []
        ssd_mount = []
        silhouette_mount = []

        for n_clusters in tqdm.tqdm(range_n_clusters):
            cluster = KMeans(
                n_clusters=n_clusters,
                n_init="auto",
                **kwargs
            )
            cluster_labels = cluster.fit_predict(self.X)

            n_clus_mount.append(n_clusters)
            ssd_mount.append(round(cluster.inertia_, 1))
            silhouette_mount.append(
                round(silhouette_score(self.X, cluster_labels), 3),
            )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        ax1.plot(range(min_cluster, max_cluster), ssd_mount, 'o--')
        ax1.set_xlabel("K Value")
        ax1.set_ylabel("Sum of Squared Distances")

        pd.Series(ssd_mount).diff().plot(kind='bar', ax=ax2)
        ax2.set_xlabel("K Value")
        ax2.set_ylabel("Change in Sum of Squared Distances")
        plt.tight_layout()
        plt.show()

        df = pd.DataFrame({
            'n_clusters': n_clus_mount,
            'ssd': ssd_mount,
            'silhouette': silhouette_mount,
            'diff': pd.Series(ssd_mount).diff()
        })

        return df.transpose()

    def kmean_knife(
            self,
            min_cluster,
            max_cluster,
            step=1,
            knife=True,
            **kwargs
    ):

        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakages

        # ARGUMENTS:

        #     min_n - minimum amount of cluster
        #     max_n - maximum amount of cluster

        range_n_clusters = list(range(min_cluster, max_cluster, step))

        n_clus_knife = []
        ssd_knife = []
        silhouette_knife = []

        for n_clusters in tqdm.tqdm(range_n_clusters):

            try:
                cluster = KMeans(
                    n_clusters=n_clusters,
                    n_init="auto",
                    **kwargs
                )
                cluster_labels = cluster.fit_predict(self.X)

                n_clus_knife.append(n_clusters)
                ssd_knife.append(round(cluster.inertia_, 1))
                silhouette_knife.append(
                    round(silhouette_score(self.X, cluster_labels),3),
                )

                print(
                    "n_clusters =",
                    n_clusters,
                    'ssd =',
                    round(cluster.inertia_, 1),
                    "average silhouette_score =",
                    round(silhouette_score(self.X, cluster_labels),3),)

                if knife:

                    knife_show(
                        self,
                        cluster_labels,
                        n_clusters,
                    )

            except ValueError:
                continue

        df = pd.DataFrame({
            'ssd': ssd_knife,
            'silhouette': silhouette_knife,
            'diff': pd.Series(ssd_knife).diff()
            }, index=range_n_clusters)

        return df.transpose()
