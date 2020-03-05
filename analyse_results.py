import os, re, json, time, collections, errno
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pprint import pprint
from customer_behaviour.tools.cluster import Cluster
from customer_behaviour.tools.result import Result
from customer_behaviour.tools import dgm as dgm
from customer_behaviour.tools.tools import save_plt_as_eps

dir_path = '/saved_results/gail2/discrete_events/1_expert(s)/case_2/2020-03-03_13-21-20' # no sigmoid AIRL test


def main():
    result = Case2(dir_path)
    root_path = os.getcwd() + dir_path
    #fig_traj = result.plot_trajectories(n_trajectories = 1)
    
    result.plot_clusters(n_dim = 3, show_benchmark = True)
    #fig_stats = result.plot_statistics()
    #fig_stats_cluster = result.plot_cluster_data()
    try:
        os.makedirs(root_path + '/figs')
        fig_path = root_path + '/figs'
        save_plt_as_eps(fig_traj, fig_path + '/trajectories.eps')
        save_plt_as_eps(fig_stats, fig_path + '/stats.eps')
        save_plt_as_eps(fig_stats_cluster, fig_path + '/cluster_stats.eps')
        print('Figures have been saved! Take a look in ' + str(fig_path))
    except OSError as e:
        fig_path = root_path + '/figs'
        print('Figures are already saved! Take a look in ' + str(fig_path))
    

############################
########## Case 3 ##########
############################

class Case3(Result):
    # There is only one expert
    # State representation: [days elapsed since last purchase, 
    #                        days elapsed between last purchase and the purchase before that, ...]
    
    def __init__(self, dir_path):
        Result.__init__(self, dir_path)
        self.n_historical_events = self.learner_states.shape[2]
    
    def plot_trajectories(self, n_trajectories=None, expert=False):
        raise NotImplementedError
        
    def get_history(self, initial_state):
        history = []
        for x in initial_state:
            while x > 1:
                history.append(0)
                x -= 1
            history.append(1)
        history.reverse()
        return history
    
############################
########## Case 2 ##########
############################ 

class Case2(Result):
    # There is only one expert
    # State representation: [historical purchases]
    
    def __init__(self, dir_path):
        Result.__init__(self, dir_path)
        self.dir_path = dir_path
        self.n_historical_events = self.learner_states.shape[2]
        
    def plot_trajectories(self, n_trajectories=None):
        expert_states = self.expert_states[0]
        expert_actions = self.expert_actions[0]
        expert_history = expert_states[0][:]
        
        if n_trajectories is None: n_trajectories = self.n_learner_trajectories
        
        for i in range(n_trajectories):
            states = self.learner_states[i]
            actions = self.learner_actions[i]
            history = states[0][:]
            cluster = Cluster(self.learner_features[i], self.expert_features)
            self.mean_dist, self.min_dist, self.max_dist = cluster.get_dist_between_clusters()
                
            fig = self.plot(expert_history, expert_actions, history, actions)  
            
            fig.suptitle('mean_dist: %.1f, min_dist: %.1f, max_dist: %.1f, var_expert_cluster: %.1f' 
                % (self.mean_dist, self.min_dist, self.max_dist, cluster.expert_within_SS))
                         
        fig.tight_layout()
        plt.show()
        return fig

############################
########## Case 1 ##########
############################
        
class Case1(Result):
    # There is only one expert
    # State representation: [sex, age, historical purchases]
    
    def __init__(self, dir_path):
        Result.__init__(self, dir_path) 
        self.n_historical_events = self.learner_states.shape[2] - 2
            
    def plot_trajectories(self, n_trajectories=None):
        raise NotImplementedError

if __name__ == '__main__':
    main()

###########################
########## Trash ##########
###########################

'''
expert_states = self.expert_states[0]
    expert_actions = self.expert_actions[0]
    expert_sex = expert_states[0][0]
    expert_age = expert_states[0][1]
    expert_history = expert_states[0][2:]
    
    if n_trajectories is None: n_trajectories = self.n_learner_trajectories
    
    for i in range(n_trajectories):
        states = self.learner_states[i]
        actions = self.learner_actions[i]
        sex = states[0][0]
        age = states[0][1]
        history = states[0][2:]
        
        expert_sex_str = 'female' if expert_sex == 1 else 'male'
        sex_str = 'female' if sex == 1 else 'male'
        text = 'Expert: {}, {} years old | Learner: {}, {} years old'.format(
            expert_sex_str, get_age(expert_age), sex_str, get_age(age))
            
        fig = self.plot(expert_history, expert_actions, history, actions)  
        fig.suptitle('Demonstration #{}'.format(i+1))
        fig.text(0.5, 0, text, horizontalalignment='center', verticalalignment='center')
                     
    fig.tight_layout()
    plt.show()

if n_trajectories is None: 
    n_trajectories = self.n_expert_trajectories if expert else self.n_learner_trajectories
    
    for i in range(n_trajectories):
        if expert:
            states = self.expert_states[i]
            actions = self.expert_actions[i]
        else:
            states = self.learner_states[i]
            actions = self.learner_actions[i]
            cluster = Cluster(self.demo_features[i], self.expert_features)
            self.mean_dist, self.min_dist, self.max_dist = cluster.get_dist_between_clusters()
        
        history = self.get_history(states[0])
        
        fig = self.plot(history, actions, expert)
        
        fig.suptitle('mean_dist: %.1f, min_dist: %.1f, max_dist: %.1f, var_expert_cluster: %.1f' 
            % (self.mean_dist, self.min_dist, self.max_dist, cluster.expert_within_SS))
        
    plt.show()
    fig.tight_layout()      
'''
