import os
import re
import json
from customer_behaviour.tools.time_series_analysis import FeatureExtraction
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

class Result():
    def __init__(self, dir_path):
        self.expert_data = os.getcwd() + dir_path + '/eval_expert_trajectories.npz' # change to eval_expert_trajectories.npz
        self.learner_data = os.getcwd() + dir_path + '/trajectories.npz'
        self.action_probs_path = os.getcwd() + dir_path + '/action_probs.npz'
        self.scores_path = os.getcwd() + dir_path + '/scores.txt'
        self.cluster_data_path = os.getcwd() + dir_path + '/cluster.txt'
        self.args_path = os.getcwd() + dir_path + '/args.txt'

        args = self.read_json_txt(self.args_path)
        
        # self.action_probs = self.load_action_probs(self.action_probs_path) if os.path.exists(self.action_probs_path) else None
        self.expert_states, self.expert_actions = self.load_data(self.expert_data)
        self.learner_states, self.learner_actions = self.load_data(self.learner_data)

        self.n_expert_trajectories = self.expert_states.shape[0]
        #print(self.expert_states)
        #print(self.learner_states)
        #print(self.learner_states.shape)

        self.n_learner_trajectories, self.episode_length, _ = self.learner_states.shape

        self.expert_features = []
        self.learner_features = []
        for i in range(self.n_expert_trajectories):
            self.expert_features.append(self.get_features(self.expert_actions[i]))

        for j in range(self.n_learner_trajectories):
            self.learner_features.append(self.get_features(self.learner_actions[j]))

    def get_features(self, trajectory):
            tsa = FeatureExtraction(trajectory, case='discrete_events')
            return tsa.get_features()
    
    def load_data(self, file):
        data = np.load(file, allow_pickle=True)
        assert sorted(data.files) == sorted(['states', 'actions'])

        states = data['states']
        actions = data['actions']

        return states, actions

    def load_action_probs(self, path):
        data = np.load(path, allow_pickle=True)
        assert sorted(data.files) == sorted(['action_probs'])
        return data['action_probs']

    def plot_clusters(self, n_dim = 2):
        features = self.expert_features.copy()
        features.extend(self.learner_features)
        features = np.array(features)
        X_embedded = TSNE(n_components=n_dim).fit_transform(features)
        expert_cluster = X_embedded[:self.n_expert_trajectories,:]
        agent_cluster = X_embedded[self.n_expert_trajectories:,:]

        if n_dim == 2:
            plt.scatter(expert_cluster[:,0], expert_cluster[:,1])
            plt.scatter(agent_cluster[:,0], agent_cluster[:,1])
            plt.show()
        elif n_dim == 3:
            fig = pyplot.figure()
            ax = Axes3D(fig)
            ax.scatter(expert_cluster[:,0], expert_cluster[:,1], expert_cluster[:,2])
            ax.scatter(agent_cluster[:,0], agent_cluster[:,1], agent_cluster[:,2])
            pyplot.show()


    def plot_cluster_data(self):
        episodes, mean_dist, min_dist, max_dist, avg_dist_from_centroid_agent, avg_dist_from_centroid_expert = self.read_cluster_data()

        plt.subplot(2,3,1)
        plt.plot(episodes, mean_dist)
        plt.xticks(rotation=90)
        plt.xlabel('Episode')
        plt.ylabel('Mean dist. between clusters')

        plt.subplot(2,3,2)
        plt.plot(episodes, min_dist)
        plt.xticks(rotation=90)
        plt.xlabel('Episode')
        plt.ylabel('Min dist. between clusters')

        plt.subplot(2,3,3)
        plt.plot(episodes, max_dist)
        plt.xticks(rotation=90)
        plt.xlabel('Episode')
        plt.ylabel('Max dist. between clusters')

        plt.subplot(2,3,4)
        plt.plot(episodes, avg_dist_from_centroid_agent)
        plt.xticks(rotation=90)
        plt.xlabel('Episode')
        plt.ylabel("Mean dist. to centroid in agent cluster")

        plt.subplot(2,3,5)
        plt.xticks(rotation=90)
        plt.plot(episodes, avg_dist_from_centroid_expert)
        plt.xlabel('Episode')
        plt.ylabel("Mean dist. to centroid in expert cluster")

        plt.tight_layout()
        plt.show()

    def plot_statistics(self):
        discriminator_loss, policy_loss, average_rewards, episodes, value, value_loss, n_updates, average_entropy = self.read_scores_txt()

        plt.subplot(2,3,1)
        plt.plot(episodes, discriminator_loss)
        plt.xlabel('Episode')
        plt.ylabel('Average discriminator loss')

        plt.subplot(2,3,2)
        plt.plot(episodes, policy_loss)
        plt.xlabel('Episode')
        plt.ylabel('Average policy loss')

        plt.subplot(2,3,3)
        plt.plot(episodes, value)
        plt.xlabel('Episode')
        plt.ylabel('Average value')

        plt.subplot(2,3,4)
        plt.plot(episodes, average_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average reward')

        plt.subplot(2,3,5)
        plt.plot(episodes, value_loss)
        plt.xlabel('Episode')
        plt.ylabel('Average value loss')

        plt.subplot(2,3,6)
        plt.plot(episodes, average_entropy)
        plt.xlabel('Episode')
        plt.ylabel('Average entropy')
        
        plt.tight_layout()
        plt.show()

    def read_cluster_data(self):
        episodes = []
        with open(self.scores_path, 'r') as file:
            lines = file.readlines()
            for idx, line in enumerate(lines):
                if idx > 0:
                    line1 = line.split(" ")[0]
                    line2 = re.split(r'\t+', line1)
                    episode = float(line2[0])
                    episodes.append(episode)


        with open(self.cluster_data_path, 'r') as file: 
            lines = file.readlines()

        columns = []

        mean_dist = []
        min_dist = []
        max_dist = []
        avg_dist_from_centroid_agent = []
        avg_dist_from_centroid_expert = []

        for idx, line in enumerate(lines):
            if idx > 0:
                line1 = line.split(" ")[0]
                line2 = re.split(r'\t+', line1)
                line3 = [float(x.rstrip("\n\r")) for x in line2]

                mean_dist.append(line3[0])
                min_dist.append(line3[1])
                max_dist.append(line3[2])
                avg_dist_from_centroid_agent.append(line3[3])
                avg_dist_from_centroid_expert.append(line3[4])

        return episodes, mean_dist, min_dist, max_dist, avg_dist_from_centroid_agent, avg_dist_from_centroid_expert

    def read_scores_txt(self):
        file_obj = open(self.scores_path, "r") 
        lines = file_obj.readlines()
        
        columns = []
        
        discriminator_loss = []
        policy_loss = []
        average_rewards = []
        average_entropy = []
        value_loss = []
        value = []
        average_value = []
        n_updates = []
        episodes = []

        for idx, line in enumerate(lines):
            if idx == 0:
                line1 = line.split(" ")[0]
                columns = re.split(r'\t+', line1)
                columns = [x.rstrip("\n\r") for x in columns]

                i_dl = columns.index('average_discriminator_loss')
                i_pl = columns.index('average_policy_loss')
                i_r = columns.index('average_rewards')
                i_v = columns.index('average_value')
                i_vl = columns.index('average_value_loss')
                i_u = columns.index('n_updates')
                i_e = columns.index('episodes')
                i_ae = columns.index('average_entropy')

            if idx > 0: # We do not want the column names
                line1 = line.split(" ")[0]
                line2 = re.split(r'\t+', line1)
                
                discriminator_loss.append(float(line2[i_dl]))
                policy_loss.append(float(line2[i_pl].rstrip("\n\r")))     
                average_rewards.append(float(line2[i_r].rstrip("\n\r")))
                value_loss.append(float(line2[i_vl]))
                value.append(float(line2[i_v]))
                n_updates.append(float(line2[i_u].rstrip("\n\r")))
                episodes.append(float(line2[i_e].rstrip("\n\r")))
                average_entropy.append(float(line2[i_ae]))

        file_obj.close()

        return discriminator_loss, policy_loss, average_rewards, episodes, value, value_loss, n_updates, average_entropy

    def read_json_txt(self, path):
        return json.loads(open(path,"r").read())
    
    def plot(self, expert_history, expert_actions, learner_history, learner_actions):        
        fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
        
        ax11 = axes[0, 0]
        ax12 = axes[0, 1]
        ax21 = axes[1, 0]
        ax22 = axes[1, 1]

        ax11.plot(expert_history)
        ax12.plot(expert_actions)
        ax21.plot(learner_history)
        ax22.plot(learner_actions)
        
        # Set titles
        ax11.set_title("Expert's history")
        ax12.set_title("Expert's actions")
        ax21.set_title("Learner's history")
        ax22.set_title("Learner's actions")
        
        # Set x-labels
        ax21.set_xlabel('Day')
        ax22.set_xlabel('Day')
        
        # Set y-labels
        ax11.set_yticks([0, 1])
        ax11.set_yticklabels(['No purchase', 'Purchase'])
        ax21.set_yticks([0, 1])
        ax21.set_yticklabels(['No purchase', 'Purchase'])
        
        return fig

########################################
########## Helper function(s) ##########
########################################

def get_age(age):
    if age < 0.2:
        return '18-29'
    elif age < 0.4:
        return '30-39'
    elif age < 0.6:
        return '40-49'
    elif age < 0.8:
        return '50-59'
    elif age < 1.0:
        return '60-69'
    else:
        return '70-80'