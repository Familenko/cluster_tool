import pandas as pd

from agglomerative_cluster import AgglomerativeCluster
from kmean import KMeansCluster


class Cluster(AgglomerativeCluster, KMeansCluster):
    def __init__(self, X: pd.DataFrame):
        self.df: pd.DataFrame = X
        self.X: pd.DataFrame = X
