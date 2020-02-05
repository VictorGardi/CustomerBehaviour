import os
import dgm as dgm
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class User:
    def __init__(self, model, time_steps = 50, n_product_groups = 6, n_historic_events = 20, analytic_level = 'discrete_events'):
        self.time_steps = time_steps
        self.model = model.DGM()
        self.model.spawn_new_customer()    
        self.age = self.model.age.transpose()
        self.sex = self.model.sex
        self.analytic_level = analytic_level
        self.n_historic_events = n_historic_events
        self.time_series = self.model.sample(time_steps) 
        self.time_series_discrete = self.get_discrete_receipt()
        self.discrete_buying_events = self.get_discrete_buying_event()
        self.mean_freq = None
        self.var_freq = None
        self.mean_cost = None
        self.var_cost = None
        self.n_product_groups = n_product_groups
        self.set_features()
        self.get_features()
        self.save_trajectory_as_npz()
        

    def set_features(self):
        if self.sex == 1:
            self.sex_color = 'red'
        elif self.sex == 0:
            self.sex_color = 'blue'
        elif self.age < 30:
            self.age_color = 'red'
            self.value = 0
        elif 30 <= self.age < 40:
            self.age_color = 'blue'
            self.value = 1
        elif 40 <= self.age < 50:
            self.age_color = 'green'
            self.value = 2
        elif 50 <= self.age < 60:
            self.age_color = 'yellow'
            self.value = 3
        elif 60 <= self.age < 70:
            self.age_color = 'purple'
            self.value = 4
        elif 70 <= self.age:
            self.age_color = 'black'
            self.value = 5

    def get_discrete_receipt(self):
        time_series = self.time_series.copy()
        time_series[time_series > 0] = 1
        return time_series

    def get_discrete_buying_event(self):
        time_series = self.time_series.copy()
        discrete_buying_events = np.zeros((self.time_steps,))
        for i in range(self.time_steps):
            if np.sum(time_series[:,i]) > 0:
                discrete_buying_events[i] = 1
        return discrete_buying_events

    def save_trajectory_as_npz(self):
        states = list()
        actions = list()
        if self.analytic_level == 'discrete_events':
            for i in range(self.time_steps-self.n_historic_events):
                state = [self.value, self.sex]
                state.append(self.discrete_buying_events[i:self.n_historic_events+i])
                actions.append(self.discrete_buying_events[self.n_historic_events + i])
                states.append(state)
            np.savez(os.getcwd() + '/expert_trajectories.npz', states=np.array(states, dtype=object),
             actions=np.array(actions, dtype=object))                
                
    def get_features(self):
        self.mean_freq, self.std_freq = self.get_mean_std_freq()
        self.mean_cost, self.std_cost = self.get_mean_std_cost()

    def get_mean_std_freq(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the discrete time series
        indices = np.argwhere(self.time_series)
        for i in range(self.n_product_groups):
            # calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i),1])
            mean_frequencies.append(np.mean(tmp))
            std_frequencies.append(np.std(tmp))
        return np.mean(mean_frequencies), np.std(std_frequencies)

    def get_mean_std_cost(self):
        costs = list()
        indices = np.nonzero(self.time_series)
        costs = self.time_series[indices]
        return np.mean(costs), np.std(costs)


usr = User(model = dgm, time_steps = 30)
#print(usr.time_series)
#print(usr.time_series_discrete)
print(usr.discrete_buying_events)
print('Sex: ' + str(usr.sex))
print('Age: ' + str(usr.age))


# usr.get_features()
# print('mean freq: ' + str(usr.mean_freq))
# print('std freq: ' + str(usr.std_freq))
# print('mean cost: ' + str(usr.mean_cost))
# print('std cost: ' + str(usr.std_cost))

#
#plt.legend(legensd)
#plt.show()

### --- plot dependent on sex --- ###
# n_experts = 100
# costs = list()
# freqs = list()
# color = list()
# for i in range(n_experts):
#     usr = User(model=dgm, time_steps=300)
#     costs.append(usr.mean_cost)
#     freqs.append(usr.mean_freq)
#     color.append(usr.sex_color)

# plt.scatter(costs, freqs, c = color)
# plt.show()









