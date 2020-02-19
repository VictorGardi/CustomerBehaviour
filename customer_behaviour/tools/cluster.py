import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class Cluster():
    def __init__(self, agent_features, expert_features):
        self.agent_features = np.array(agent_features)
        self.expert_features = np.array(expert_features)
        self.agent_centroid = self.get_centroid(self.agent_features)
        self.expert_centroid = self.get_centroid(self.expert_features)
        self.agent_within_SS = self.get_within_cluster_SS(self.agent_features, self.agent_centroid)
        self.expert_within_SS = self.get_within_cluster_SS(self.expert_features, self.expert_centroid)
        #self.mean_dist, self.min_dist, self.max_dist = self.get_dist_between_clusters(self.agent_features, self.expert_features)
        
        # plt.scatter(self.agent_features[:,0], self.agent_features[:,1])
        # plt.scatter(self.expert_features[:,0], self.expert_features[:,1])
        # plt.scatter(self.agent_centroid[0], self.agent_centroid[1])
        # plt.scatter(self.expert_centroid[0], self.expert_centroid[1])
        # plt.show()

    def get_centroid(self, features):
        features = np.array(features)
        return np.mean(features, axis=0)

    def get_within_cluster_SS(self, cluster, centroid):
        cluster = np.array(cluster)
        SS = 0
        for i in range(cluster.shape[0]):
            SS += (np.linalg.norm(cluster[i,:]-centroid))
        return SS

    def get_dist_between_clusters(self):
        current_min_dist = 100000
        current_max_dist = 0
        for i in range(self.agent_features.shape[0]):
            dist = distance.cdist(self.agent_features[i,:].reshape((1,-1)), self.expert_features, 'euclidean')

            if np.min(dist) < current_min_dist:
                current_min_dist = np.min(dist)
            if np.max(dist) > current_max_dist:
                current_max_dist = np.max(dist)
        mean_dist = np.linalg.norm(self.expert_centroid-self.agent_centroid)
        return mean_dist, current_min_dist, current_max_dist


# ls = [1,5] 
# ls1 = [1.5,4]
# ls2 = [4,2]
# ls3 = [5,1]

# age = []
# exp = []
# age.append(ls)
# #age.append(ls1)
# exp.append(ls2)
# exp.append(ls3)
# cluster = Cluster(age, exp)
# print(cluster.mean_dist)
# print(cluster.min_dist)
# print(cluster.max_dist)