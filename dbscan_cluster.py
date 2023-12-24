import numpy as np
import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from knife import knife_show


class DBSCANCluster:
    def dbscan_knife(
            self,
            mod="eps",
            min_eps=0.01,
            max_eps=1,
            range_eps=10,
            min_sample=1,
            max_sample=5,
            epsindot=0.5,
            dotineps=5,
            knife=True
    ):
        # DESCRIPTION:

        #     Build selected range of clusters and represent knife metric to ich of them
        #     'Knifes' should be similar to ich other and have good shape without leakeges

        # ARGUMENTS:

        #     mod - different way for assessment density depends on eps or dot
        #     min_eps - minimum parameter for eps in case of use mod='eps'
        #     max_eps - maximum parameter for eps in case of use mod='eps'
        #     range_eps - amount of eps parameters tested for eps in case of use mod='eps'
        #     min_sample - minimum amount of dots in area in case of use mod='dot'
        #     max_sample - maximum amount of dots in area in case of use mod='dot'
        #     epsindot - static parameter for eps in case of use mod='dot'
        #     dotineps - static parameter for dot in case of use mod='eps'

        check_values = []
        silhouette = []
        self.outlier_percent = []
        amount_of_clusters = []

        if mod == 'eps':
            self.xmin = min_eps
            self.xmax = max_eps
            self.check_range = np.linspace(min_eps, max_eps, range_eps)

        if mod == 'dot':
            self.xmin = min_sample
            self.xmax = max_sample
            self.check_range = range(min_sample, max_sample + 1)

        for check_n in tqdm(self.check_range):
            fig, ax1 = plt.subplots(1, 1)
            fig.set_size_inches(7, 4)

            if mod == 'eps':
                cluster = DBSCAN(eps=check_n, min_samples=dotineps)

            if mod == 'dot':
                cluster = DBSCAN(min_samples=check_n, eps=epsindot)

            try:

                cluster_labels = cluster.fit_predict(self.X)

                perc_outliers = (
                        100 * np.sum(cluster.labels_ == -1) / len(cluster.labels_)
                )
                self.outlier_percent.append(perc_outliers)

                amount_of_clusters.append(str(len(set(cluster_labels))))
                check_values.append(str(check_n))
                silhouette.append(
                    str(round(silhouette_score(self.X, cluster_labels), 3), )
                )

                print(
                    "check_n =",
                    check_n,
                    "average silhouette_score =",
                    round(silhouette_score(self.X, cluster_labels), 3),
                    'outliers =',
                    perc_outliers
                )

                if knife:

                    knife_show(
                        self,
                        cluster_labels,
                        check_n,
                        ax1
                    )

            except ValueError:
                continue

        df = pd.DataFrame({
            'n_clusters': amount_of_clusters,
            'silhouette': silhouette,
            'outliers': self.outlier_percent,
        }, index=self.check_range)

        return df.transpose()

    def dbscan_outliers(self, percent=1):

        # DESCRIPTION:

        #     Build diagram to show amount of outliers based on
        #     tested parameter (eps or dot) in knife method

        # ARGUMENTS:

        #     percent - build hlines on diagram for better visualization

        sns.lineplot(x=self.check_range, y=self.outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Check Value")
        plt.hlines(y=percent, xmin=self.xmin, xmax=self.xmax, colors='red', ls='--')
        plt.grid(alpha=0.2)

        plt.show()
