import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from clusters_class.agglomerative_cluster import AgglomerativeCluster
from clusters_class.kmean_cluster import KMeansCluster
from clusters_class.dbscan_cluster import DBSCANCluster


class Cluster(AgglomerativeCluster, KMeansCluster, DBSCANCluster):
    cluster_dic = {'agglo': AgglomerativeClustering(),
                   'kmean': KMeans(),
                   'dbscan': DBSCAN()}

    scaler_dic = {'StandardScaler': StandardScaler(),
                  'MinMaxScaler': MinMaxScaler()}

    cluster = None
    pipe = None
    scaler = None
    result_df = None

    def __init__(self, X: pd.DataFrame):
        self.df: pd.DataFrame = X
        self.X: pd.DataFrame = X

    def preprocessing(self, scaler='StandardScaler'):

        # DESCRIPTION:

        #     Preprocess the data with MinMaxScaler & StandardScaler

        self.scaler = self.scaler_dic[scaler]
        scaled_data = self.scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data)

    def build_cluster(self, cluster='kmean', **kwargs):

        # DESCRIPTION:

        #     Build selected amount of cluster

        self.cluster = self.cluster_dic[cluster](**kwargs)
        cluster_labels = self.cluster.fit_predict(self.X)
        self.result_df = self.df
        self.result_df['cluster'] = cluster_labels

    def build_pipe(self):
        self.pipe = make_pipeline(self.scaler, self.cluster)
        return self.pipe

    def simple_check(self, mode='built', target='cluster', alpha=0.5):

        # DESCRIPTION:

        #     Check data with PCA preprocessing including DataFrame
        #     built in build_knife and build_simple method

        # ARGUMENTS:

        #     mode - tested mode, data from which DataFrame will be taken
        #     target - hue for scatterplot diagram in case of using cluster = 'origin'
        #     alpha - alpha for scatterplot

        if mode == 'origin':
            X = self.df
            df = self.df
            target = self.df[target]

        elif mode == 'built':
            X = self.result_df
            df = self.result_df

        elif mode == "outliers":
            X = self.result_df
            df = self.result_df
            target = np.where(df['cluster'] == -1, 'outliers', 'normal')

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        X = pd.DataFrame(principal_components)

        print(f'pca.explained_variance_ratio_ = '
              f'{pca.explained_variance_ratio_}')
        print(f'np.sum(pca.explained_variance_ratio_ = '
              f'{np.sum(pca.explained_variance_ratio_)}')

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X[0], y=X[1], data=df, hue=target, alpha=alpha)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def after_pie(self, bins=20):

        # DESCRIPTION:

        #     Build distribution diagram

        if len(self.result_df['cluster'].value_counts()) < 10:
            cluster_counts = self.result_df['cluster'].value_counts()
            plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')

        else:
            sns.displot(data=self.result_df, x='cluster', kde=True, color='green', bins=bins)

        plt.show()

    def after_heat(self, sh=12, vi=4):

        # DESCRIPTION:

        #     Build feature correlation diagram for 'knife' or 'simple' algorithm

        # ARGUMENTS:

        #     cluster - tested mode
        #     sh - height of diagram
        #     vi - length of diagram

        cat_means = self.result_df.groupby('cluster').mean()

        scaler = MinMaxScaler()
        data = scaler.fit_transform(cat_means)
        scaled_means = pd.DataFrame(data, cat_means.index, cat_means.columns)

        if scaled_means.reset_index().iloc[0]['cluster'] == -1:
            ax = plt.figure(figsize=(sh,vi), dpi=200)
            ax = sns.heatmap(scaled_means.iloc[1:], annot=True, cmap='Greens')

        else:
            ax = plt.figure(figsize=(sh, vi), dpi=200)
            ax = sns.heatmap(scaled_means, annot=True, cmap='Greens')

        return ax

    def dendrogram(self, n_clusters=2):

        # DESCRIPTION:

        #     Check the optimal amount of cluster by scipy.cluster.hierarchy
        #     By using this diagram, it is possible to make an assessment
        #     of the chosen amount of clusters on actual data

        Z = linkage(self.X, 'ward')

        plt.figure(figsize=(10, 8))
        dendrogram(Z, color_threshold=np.sqrt(len(self.X.columns)))
        plt.xticks(rotation=90);
        plt.title('Automatic Clustering')
        plt.show()

        # Plot dendrogram with specified number of clusters
        plt.figure(figsize=(10, 8))
        plt.title('Custom Clustering')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        dendrogram(
            Z,
            leaf_rotation=90,
            leaf_font_size=8.,
            labels=np.arange(1, len(self.X) + 1),
            color_threshold=Z[-n_clusters + 1, 2]
        )
        plt.axhline(y=Z[-n_clusters + 1, 2], color='r', linestyle='--')
        plt.xticks(rotation=90)
        plt.show()
