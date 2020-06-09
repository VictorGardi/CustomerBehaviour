import numpy as np
import matplotlib.pyplot as plt
from customer_behaviour.tools.dgm import DGM

class Case22():  # dummy encoding (dynamic)
    def __init__(self, model, n_experts=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.n_experts + n_historical_events) 

        action_space = spaces.Discrete(2)

        return observation_space, action_space

    def get_sample(self, n_demos_per_expert, n_historical_events, n_time_steps):
        temp_sample = self.model.sample(n_demos_per_expert * (n_historical_events + n_time_steps))

        sample = []
        for subsample in np.split(temp_sample, n_demos_per_expert, axis=1):
            history = subsample[:, :n_historical_events]
            data = subsample[:, n_historical_events:]
            sample.append((history, data))
        return sample

    def get_action(self, receipt):
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = state[self.n_experts:]
        new_state = [*dummy, *history[1:], action]
        return new_state

def get_states_actions(case, model, seed, episode_length = 256, n_historical_events = 96, save_for_visualisation=False):
    states = []
    actions = []

    model.spawn_new_customer(seed)
    sample = case.get_sample(1, n_historical_events, episode_length)

    for subsample in sample:
        temp_states = []
        temp_actions = []

        history = subsample[0]
        data = subsample[1]

        initial_state = case.get_initial_state(history, seed)

        state = initial_state

        temp_states.append(np.array(initial_state))  # the function step(action) returns the state as an np.array

        for i, receipt in enumerate(data.T, start=1):  # transpose since the receipts are columns in data
            action = case.get_action(receipt)
            temp_actions.append(action)

            if i == data.shape[1]:
                # The number of states and actions must be equal
                pass
            else:
                state = step(case, state, action)
                temp_states.append(state)

        states.append(temp_states)
        actions.append(temp_actions)
        if save_for_visualisation:
            np.savez('stable-baselines-test/eval_expert_trajectories.npz', states=states, actions=actions)

    return states, actions

def step(case, state, action):        
    new_state = case.get_step(state, action)
    state = new_state
    return np.array(state)

def generate_purchase_days(sample_length, bins, freq_probs, purchase_histo):
    n_purchases = np.random.choice(bins, size=1, p=freq_probs)
    purchase_days = np.random.choice(range(sample_length), size=int(n_purchases), replace=False, p=purchase_histo)

    return sorted(purchase_days)
        
def get_freqs_probs(ratios, sample_length):
    freq_probs, bins = np.histogram([ratio*sample_length for ratio in ratios], density=True, bins=20)
    bins = (bins[1:] + bins[:-1])/2
    freq_probs = freq_probs/np.sum(freq_probs)
    return freq_probs, bins

def get_purchase_probs(indices, sample_length):
    histo = np.histogram(indices, density=False, bins=sample_length)
    #histo = plt.hist(indices, density=False, bins=sample_length)
    #plt.ylabel('Purchases')
    #plt.xlabel('Days')
    #plt.show()
    return histo[0]/len(indices)

def get_purchase_ratio(sequence):
    return np.count_nonzero(sequence)/len(sequence)

def main():

    import custom_gym
    import gym
    import numpy as np
    import itertools
    import os, sys
    from os.path import join
    import pandas as pd
    tools_path = join(os.getcwd(), 'customer_behaviour/tools')
    sys.path.insert(1, tools_path)
    import policy_evaluation as pe
    import results2 as res
    from scipy.stats import wasserstein_distance as wd


    sample_length = 10000
    n_experts = 10
    n_customers = 50
    tot_customers = n_experts + n_customers
    n_last_days = 7
    model = DGM()
    case = Case22(model=model, n_experts=tot_customers)
    states = []
    actions = []
    for expert in range(tot_customers):
        state, action = get_states_actions(case = case, model=model, seed=expert, episode_length = sample_length, n_historical_events = 10, save_for_visualisation=False)
        states.append(state)
        actions.append(action)
    states = np.array(states)
    actions = np.array(actions)

    customers = res.get_distribs(states, actions)
    experts_ts = actions[:n_experts]
    new_customers_dis = customers[-n_customers:]
    experts_dis = customers[:n_experts]

    dist = []
    final_dist = []
    for i in range(n_customers):
        c = new_customers_dis[i]
        distances = [wd(c, e) for e in experts_dis]
        dist.append(min(distances))
        dummy = np.argsort(distances)[0]
        expert_ts = experts_ts[dummy,:].tolist()[0]

        length_subsample = int(sample_length/10)
        samples = [expert_ts[x:x+length_subsample] for x in range(0, len(expert_ts), length_subsample)]
        ratios = [get_purchase_ratio(sample) for sample in samples]

        freq_probs, bins = get_freqs_probs(ratios, length_subsample)   
        purchase_days = []
        for sample in samples:
            purchase_days.extend(np.nonzero(sample*10)[0].tolist())

        #purchase_days = [np.nonzero(sample)[0].tolist() for sample in samples]
        #print(purchase_days[0])
        purchase_histo = get_purchase_probs(purchase_days, sample_length)
        purchase_days = generate_purchase_days(sample_length, bins, freq_probs, purchase_histo)
        c_actions = np.zeros((sample_length,)).astype(int)
        for day in purchase_days:
            c_actions[day] = 1

        temp = c_actions.tolist()
        c_states = []
        for x in range(len(temp)-10):
            c_states.append(temp[x:x+10])
        c_states = np.asarray(c_states)
        n_actions = len(temp)
        n_states = c_states.shape[0]
        c_actions = c_actions[:n_states]
        c_dis = pe.get_distrib(c_states, c_actions)
        #c_dis= res.get_distribs(c_states, c_actions)

        final_dist.append(wd(c_dis, experts_dis[dummy]))
    print(np.mean(final_dist))



main()
