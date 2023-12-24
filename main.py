import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from agglomerative_cluster import AgglomerativeCluster
from kmean_cluster import KMeansCluster
from dbscan_cluster import DBSCANCluster


class Cluster(AgglomerativeCluster, KMeansCluster, DBSCANCluster):
    def __init__(self, X: pd.DataFrame):
        self.df: pd.DataFrame = X
        self.X: pd.DataFrame = X

    def preprocessing(self, scaler='StandardScaler'):

        # DESCRIPTION:

        #     Preprocess the data with MinMaxScaler & StandardScaler

        if scaler == 'StandardScaler':
            scaler = StandardScaler()
        if scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(self.X)
        self.X = pd.DataFrame(scaled_data)

        self.scaler = scaler
