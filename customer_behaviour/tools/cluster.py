import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Cluster():
    def __init__(self, agent_features, expert_features):

        self.agent_features = np.array(agent_features)
        self.expert_features = np.array(expert_features)

        if self.agent_features.ndim == 1:
            n_features = self.agent_features.size
            self.agent_features = np.reshape(self.agent_features, (1, n_features))

        self.agent_centroid = self.get_centroid(self.agent_features)
        self.expert_centroid = self.get_centroid(self.expert_features)
        
        self.agent_within_SS = self.get_within_cluster_SS(self.agent_features, self.agent_centroid)
        self.expert_within_SS = self.get_within_cluster_SS(self.expert_features, self.expert_centroid)
        
        # self.mean_dist, self.min_dist, self.max_dist = self.get_dist_between_clusters(self.agent_features, self.expert_features)


    def get_centroid(self, features, expert=False):
        return np.mean(features, axis=0)


    def get_within_cluster_SS(self, cluster, centroid):
        cluster = np.array(cluster)
        SS = 0
        for i in range(cluster.shape[0]):
            SS += (np.linalg.norm(cluster[i,:]-centroid))
        return SS / cluster.shape[0]

    def get_dist_between_clusters(self):
        dist = distance.cdist(self.expert_features, self.agent_features, 'euclidean')

        min_dist = np.min(dist)
        max_dist = np.max(dist)
        mean_dist = np.linalg.norm(self.expert_centroid-self.agent_centroid)
        return mean_dist, min_dist, max_dist


