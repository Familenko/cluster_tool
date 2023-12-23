import pandas as pd

from agglomerative_cluster import AgglomerativeCluster


class Cluster(AgglomerativeCluster):
    def __init__(self, X: pd.DataFrame):
        self.df: pd.DataFrame = X
        self.X: pd.DataFrame = X
