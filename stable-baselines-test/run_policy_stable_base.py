import os
import gym
import custom_gym
import chainer
import itertools
import numpy
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from math import floor
from scipy.stats import chisquare
from customer_behaviour.tools import dgm as dgm

directory = 'stable-baselines-test/'  # 2.1
episode_length = 10000
def main():
    env = gym.make('discrete-buying-events-v0')
    env.initialize_environment(case = 2, n_historical_events = 96, episode_length = episode_length, n_demos_per_expert=1, n_expert_time_steps=256, agent_seed=0)
    env = DummyVecEnv([lambda: env])
    agent = GAIL.load(os.getcwd() + '/stable-baselines-test/gail')
    
    save_agent_demo(env, agent, 'stable-baselines-test', episode_length)
    data = np.load(file, allow_pickle=True)
    assert sorted(data.files) == sorted(['states', 'actions'])

    agent_states = data['states']
    agent_actions = data['actions']

    # get expert s-a
    model = dgm.DGM()
    case = Case21(model=model)
    expert_states, expert_actions = get_states_actions(case = case, model=model, seed=0, episode_length = episode_length, n_historical_events = 96)

    # Get conditional validation states
    agent_purchase, agent_no_purchase = get_cond_val_states([agent_states], [agent_actions], n_last_days)
    expert_purchase, expert_no_purchase = get_cond_val_states(expert_states, expert_actions, n_last_days)

    # Reduce the dimensionality by treating all validation states with more than x purchases as one single state
    expert_purchase = reduce_dimensionality(expert_purchase, max_n_purchases_per_n_last_days)
    expert_no_purchase = reduce_dimensionality(expert_no_purchase, max_n_purchases_per_n_last_days)
    agent_purchase = reduce_dimensionality(agent_purchase, max_n_purchases_per_n_last_days)
    agent_no_purchase = reduce_dimensionality(agent_no_purchase, max_n_purchases_per_n_last_days)

    # Get possible validation states
    possible_val_states = [list(x) for x in itertools.product([0, 1], repeat=n_last_days)]
    possible_val_states = reduce_dimensionality(possible_val_states, max_n_purchases_per_n_last_days, True)
    possible_val_states = sort_possible_val_states(possible_val_states)

    # Get counts
    expert_counts_purchase = get_counts(expert_purchase, possible_val_states, normalize=False)
    expert_counts_no_purchase = get_counts(expert_no_purchase, possible_val_states, normalize=False)
    agent_counts_purchase = get_counts(agent_purchase, possible_val_states, normalize=False)
    agent_counts_no_purchase = get_counts(agent_no_purchase, possible_val_states, normalize=False)

    # Calculate one-way chi-square tests (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
    # Null hypothesis: The categorical data has the given frequencies
    # "Assuming that the null hypothesis is true, 
    # the p-value is the probability of obtaining agent counts as extreme as the agent counts actually observed during this test."
    # A typical rule is that all of the observed and expected frequencies should be at least 5

    # assert min(agent_counts_purchase) >= 5
    # assert min(agent_counts_no_purchase) >= 5
    # assert min(expert_counts_purchase) >=5
    # assert min(expert_counts_no_purchase) >= 5

    _, p_value_purchase = chisquare(f_obs=agent_counts_purchase, f_exp=expert_counts_purchase, ddof=0)
    print('P-value given purchase: %.5f' % p_value_purchase)
    if p_value_purchase <= alpha: print('Rejecting null hypothesis')
    _, p_value_no_purchase = chisquare(f_obs=agent_counts_no_purchase, f_exp=expert_counts_no_purchase, ddof=0)
    print('P-value given no purchase: %.5f' % p_value_no_purchase)
    if p_value_no_purchase <= alpha: print('Rejecting null hypothesis')

    x = range(len(possible_val_states))

    # Plot expert
    fig, ax = plt.subplots()
    ax.bar(x, expert_counts_no_purchase)
    set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle('Expert')
    ax.set_title('No purchase today')

    fig, ax = plt.subplots()
    ax.bar(x, expert_counts_purchase)
    set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle('Expert')
    ax.set_title('Purchase today')

    # Plot agent
    fig, ax = plt.subplots()
    ax.bar(x, agent_counts_no_purchase)
    set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle('Agent | p-value: %.5f' % p_value_no_purchase)
    ax.set_title('No purchase today')
    
    fig, ax = plt.subplots()
    ax.bar(x, agent_counts_purchase)
    set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle('Agent | p-value: %.5f' % p_value_purchase)
    ax.set_title('Purchase today')

    plt.show()

############################
##### Helper functions #####
############################

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

        initial_state = case.get_initial_state(history)

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


class Case21():
    def __init__(self, model):
        self.model = model

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(n_historical_events)

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
        action = 1 if np.any(np.nonzero(receipt)) else 0
        return action

    def get_initial_state(self, history):
        temp = history.copy()
        temp = np.sum(temp, axis=0)

        temp[temp > 0] = 1

        initial_state = temp

        return initial_state

    def get_step(self, state, action):
        new_state = [*state[1:], action]
        return new_state

def sort_possible_val_states(possible_val_states):
    temp = possible_val_states.copy()
    # temp.sort(key=lambda x: sum(x))
    # temp = np.array(temp)
    temp_splitted = []
    s = 0
    while sum(len(x) for x in temp_splitted) < len(temp):
        indices = np.argwhere(np.sum(temp, axis=1) == s)
        t = [temp[i[0]] for i in indices]
        t.sort(reverse=True)
        temp_splitted.append(t)
        s += 1
    return [item for sublist in temp_splitted for item in sublist]

def set_xticks(ax, possible_val_states, n):
    ticks = list(range(len(possible_val_states)))
    labels = []

    for x in possible_val_states:
        if sum(x) == len(x):
            labels.append('> %d purchases' % n)
        else:
            temp = [str(i) for i in x]
            temp = ', '.join(temp)
            temp = '[' + temp + ']'
            labels.append(temp)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_tick_params(rotation=90)

def reduce_dimensionality(val_states, max_n_purchases, keep_only_unique=False):
    '''
    val_states: [[v_{0}], [v_{1}], ..., [v_{n-2}], [v_{n-1}]]
    max_n_purchases: The maximum number of purchases that is allowed in a validation state
    '''
    indices = np.argwhere(np.sum(val_states, axis=1) > max_n_purchases)  # list of lists
    indices = [x[0] for x in indices]

    assert len(val_states) > 0
    n = len(val_states[0])
    substitute = n * [1]

    for i in indices:
        val_states[i] = substitute

    if keep_only_unique:
        temp = set(tuple(x) for x in val_states)
        return [list(x) for x in temp]
    else:
        return val_states

def get_cond_val_states(states, actions, n):
    n_trajectories = len(states)

    purchase = []
    no_purchase = []

    for i in range(n_trajectories):
        for temp_state, temp_action in zip(states[i], actions[i]):
            # Extract the last n days
            last_n_days = temp_state[-n:]
            val_state = [int(x) for x in last_n_days]
            # val_state.append(temp_action)
            if temp_action == 1:
                purchase.append(val_state)
            else:
                no_purchase.append(val_state)

    return purchase, no_purchase

def get_counts(observed_val_states, possible_val_states, normalize=False):
    counts = len(possible_val_states) * [0]
    for temp in observed_val_states:
        i = possible_val_states.index(temp)
        counts[i] += 1
    if normalize:
        counts = list(np.array(counts) / np.sum(counts))
    return counts

def autocorr(x, t = 20):
    result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    temp = result[floor(result.size/2):floor(result.size/2)+t]
    return temp / np.amax(temp)

############################
############################
############################

if __name__ == '__main__':
    main()