import os
import re
import json
from customer_behaviour.tools.time_series_analysis import FeatureExtraction
import numpy as np
import matplotlib.pyplot as plt

class Result():
    def __init__(self, dir_path):
        self.expert_data = os.getcwd() + dir_path + '/expert_trajectories.npz' # change to eval_expert_trajectories.npz
        self.learner_data = os.getcwd() + dir_path + '/trajectories.npz'
        self.action_probs_path = os.getcwd() + dir_path + '/action_probs.npz'
        self.scores_path = os.getcwd() + dir_path + '/scores.txt'
        self.args_path = os.getcwd() + dir_path + '/args.txt'

        args = self.read_json_txt(self.args_path)
        
        # self.action_probs = self.load_action_probs(self.action_probs_path) if os.path.exists(self.action_probs_path) else None
        self.expert_states, self.expert_actions = self.load_data(self.expert_data)
        self.learner_states, self.learner_actions = self.load_data(self.learner_data) 
        self.n_expert_trajectories = self.expert_states.shape[0]
        self.n_learner_trajectories, self.episode_length, _ = self.learner_states.shape
        self.n_historical_events = self.learner_states.shape[2]

        self.expert_features = []
        self.demo_features = []
        for i in range(self.n_expert_trajectories):
            self.expert_features.append(self.get_features(self.expert_actions[i]))

        for j in range(self.n_learner_trajectories):
            self.demo_features.append(self.get_features(self.learner_actions[j]))

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


    def plot_loss(self):
        discriminator_loss, policy_loss, average_rewards, episodes = self.read_scores_txt()
        plt.subplot(1,3,1)
        plt.plot(episodes, discriminator_loss, label='discriminator loss')
        plt.subplot(1,3,2)
        plt.plot(episodes, policy_loss, label='policy loss')
        plt.subplot(1,3,3)
        plt.plot(episodes, average_rewards, label='reward')
        plt.xlabel('Episode')
        #plt.legend()
        plt.show()

    def read_scores_txt(self):
        file_obj = open(self.scores_path,"r") 
        lines = file_obj.readlines()
        discriminator_loss = []
        policy_loss = []
        average_rewards = []
        episodes = []
        for idx, line in enumerate(lines):
            if idx > 0: # We do not want the column names
                line1 = line.split(" ")[0]
                line2 = re.split(r'\t+', line1)
                discriminator_loss.append(float(line2[8]))
                policy_loss.append(float(line2[-4].rstrip("\n\r")))
                average_rewards.append(float(line2[9].rstrip("\n\r")))
                episodes.append(float(line2[1].rstrip("\n\r")))

        return discriminator_loss, policy_loss, average_rewards, episodes

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