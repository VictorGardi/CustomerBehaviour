
from customer_behaviour import analyze_results

class Cluster():
    def __init__(self, agent_features, expert_features):
        self.agent_features = agent_features
        self.expert_features
        
    def get_centroid(self, features):
        features = np.array(features)
        return np.mean(features, axis=0)
                
    def get_dist_between_clusters(self, expert_features, agent_features):
        expert_centroid = self.get_centroid(expert_features)
        agent_centroid = self.get_centroid(agent_features)
        return np.linalg.norm(expert_centroid-agent_centroid)