import os
from CustomerBehaviour.tools import dgm as dgm
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class User:
    def __init__(self, model = dgm, time_steps = 50, n_product_groups = 6, n_historic_events = 7, max_age_cat = 5):
        self.time_steps = time_steps
        self.model = model.DGM()
        self.model.spawn_new_customer()    
        self.age = self.model.age.transpose()
        self.sex = self.model.sex
        self.n_historic_events = n_historic_events
        self.time_series = self.model.sample(time_steps) 
        self.time_series_discrete = self.get_discrete_receipt()
        self.discrete_buying_events = self.get_discrete_buying_event()
        self.max_age_cat = max_age_cat
        self.n_product_groups = n_product_groups
        self.set_features()
        

    def set_features(self):
        if self.sex == 1:
            self.sex_color = 'red'
        elif self.sex == 0:
            self.sex_color = 'blue'

        # Put each user in an age category and give that category a color
        if self.age < 30:
            self.age_color = 'red'
            self.age_cat = 0
        elif 30 <= self.age < 40:
            self.age_color = 'blue'
            self.age_cat = 1
        elif 40 <= self.age < 50:
            self.age_color = 'green'
            self.age_cat = 2
        elif 50 <= self.age < 60:
            self.age_color = 'yellow'
            self.age_cat = 3
        elif 60 <= self.age < 70:
            self.age_color = 'purple'
            self.age_cat = 4
        elif 70 <= self.age:
            self.age_color = 'black'
            self.age_cat = 5

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

    def generate_trajectory(self):
        states = list()
        actions = list()
        for i in range(self.time_steps-self.n_historic_events):
            state = [self.sex, self.age_cat / self.max_age_cat, *self.discrete_buying_events[i:self.n_historic_events+i]]
            actions.append(self.discrete_buying_events[self.n_historic_events + i])
            states.append(state)
        return states, actions
           
def main(n_experts = 1):

    demo_states = []
    demo_actions = []

    for i in range(n_experts):
        usr = User(model = dgm, time_steps = 128)
        states, actions = usr.generate_trajectory()
        demo_states.append(states)
        demo_actions.append(actions)

    print(demo_states)
    print(demo_actions)

    np.savez(os.getcwd() + '/expert_trajectories.npz', states=np.array(demo_states, dtype=object),
             actions=np.array(demo_actions, dtype=object))        



if __name__ == '__main__':
    main()











