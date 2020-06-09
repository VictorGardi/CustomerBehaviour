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


###########################################
###########################################

def main():

    import custom_gym
    import gym
    from customer_behaviour.tools.dgm import DGM
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
    import seaborn as sns


    sample_length = 10000
    expert_ls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]
    n_customers = 20
    n_runs = 3
    tot_customers = max(expert_ls) + n_customers
    n_last_days = 7
    model = DGM()
    case = Case22(model=model, n_experts=tot_customers)
    states = []
    actions = []
    for expert in range(tot_customers):
        state, action = get_states_actions(case = case, model=model, seed=expert, episode_length = sample_length, n_historical_events = 1000, save_for_visualisation=False)
        states.append(state)
        actions.append(action)
    states = np.array(states)
    actions = np.array(actions)

    customers = res.get_distribs(states, actions)

    data = []
    for n_experts in expert_ls:
        #expert_states = states[:n_experts]
        #expert_actions = actions[:n_experts]
        for _ in range(n_runs):

            #experts = np.random.choice(customers[:max(expert_ls)], n_experts)
            choice_indices = np.random.choice(range(len(customers)- n_customers), n_experts, replace=False)
            experts = [customers[i] for i in choice_indices]
            #experts = customers[:n_experts]

            #avg_expert = pe.get_distrib(expert_states, expert_actions)
            new_customers = customers[-n_customers:]
            dist = []
            for i in range(n_customers):
                c = new_customers[i]
                distances = [wd(c, e) for e in experts]
                dist.append(min(distances))
            data.append([n_experts, np.mean(dist)])
        
    df = pd.DataFrame(data, columns=['Number of experts', 'Average distance to closest customer'])
    sns.set(style='darkgrid')
    
    g = sns.relplot(x='Number of experts', y='Average distance to closest customer', hue=None, ci=95, kind='line', data=df, facet_kws={'legend_out': False})
    plt.show()


    #plt.plot(expert_ls, mean_distances, color="black")
    #plt.grid()
    #plt.xlabel('Number of customers')
    #plt.ylabel('Average distance to closest customer')
    #plt.show()


if __name__ == '__main__':
    main()

