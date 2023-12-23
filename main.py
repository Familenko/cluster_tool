import numpy as np
import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


class Clusterer():

    def __init__(self, X):
        self.df = X
        self.X = X

    def agglo_distance(self, min_d=1, max_d=5, range_d=10, save=None):

        # DESCRIPTION:

        #     Build diagram with distance correlation of data in different cluster

        # ARGUMENTS:

        #     min_d - minimum level of distance
        #     max_d - maximum level of distance
        #     range_d - amount of tested distances

        distance_threshold_list = list(np.linspace(min_d, max_d, range_d))

        n_claster_dis = []
        distance_threshold = []
        silhouette_dis = []

        for distance_threshold_n in tqdm(distance_threshold_list, desc="Checking distance"):
            clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold_n)
            cluster_labels = clusterer.fit_predict(self.X)

            distance_threshold.append(distance_threshold_n)
            n_claster_dis.append(len(np.unique(cluster_labels)))
            silhouette_dis.append(round(silhouette_score(self.X, cluster_labels), 3), )

        plt.figure(figsize=(12, 5), dpi=200)
        sns.scatterplot(x=n_claster_dis, y=distance_threshold, hue=silhouette_dis, s=100)
        plt.title("Distance plot")
        plt.xlabel("Number of claster")
        plt.ylabel("Distance_threshold")
        plt.xticks(range(min(n_claster_dis), max(n_claster_dis) + 1))
        plt.yticks(np.arange(0, max(distance_threshold) + 1, step=1))
        plt.legend(title='Silhouette', loc='upper right')

        if save == 'save':
            plt.savefig('my_plot.png')

        plt.show()

        df = pd.DataFrame({
            'n_clusters': n_claster_dis,
            'distance_threshold': distance_threshold,
            'silhouette': silhouette_dis
        })

        return df.transpose()
