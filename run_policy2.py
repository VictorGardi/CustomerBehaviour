import os
import gym
import custom_gym
import chainer
import itertools
import chainerrl
import numpy
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from main import A3CFFSoftmax
from math import floor
from chainerrl.misc.batch_states import batch_states
from scipy.stats import chisquare

directory = 'saved_results/gail/discrete_events/1_expert(s)/case_21/monday_0302/2020-03-02_19-42-48'  # 2.1
# directory = 'saved_results/gail/discrete_events/1_expert(s)/case_2/friday_0228/2020-02-29_06-53-54'  # 2.0

sample_length = 10000
n_experts = 1
n_last_days = 7
max_n_purchases_per_n_last_days = 2
alpha = 0.01  # significance level

def main():
    # Load parameters and settings
    args_path = join(directory, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    # Create environment
    env = gym.make('discrete-buying-events-v0')
    env.initialize_environment(
        case=args['state_rep'], 
        n_historical_events=args['n_historical_events'], 
        episode_length=sample_length,  # length of the agent's sample
        n_demos_per_expert=1,
        n_expert_time_steps=sample_length,  # length of expert's sample
        agent_seed=args['agent_seed']
        )

    # Initialize model and observation normalizer
    model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=(64, 64))  # Assmuning state = [historical purchases]
    obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], clip_threshold=5)

    # Load model and observation normalizer
    model_directory = [x[0] for x in os.walk(directory)][1]
    chainer.serializers.load_npz(join(model_directory, 'model.npz'), model)
    chainer.serializers.load_npz(join(model_directory, 'obs_normalizer.npz'), obs_normalizer)

    # Sample agent data
    agent_states, agent_actions = sample_from_policy(env, model, obs_normalizer)

    # Sample expert data
    trajectories = env.generate_expert_trajectories(
        n_experts=n_experts, 
        out_dir='.',
        seed=True, 
        eval=False
        )
    expert_states = trajectories['states']
    expert_actions = trajectories['actions']

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

def sample_from_policy(env, model, obs_normalizer):
    xp = numpy
    phi = lambda x: x

    states = []
    actions = []

    obs = env.reset().astype('float32')  # Initial state
    states.append(obs)
    done = False
    while not done:
        b_state = batch_states([obs], xp, phi)
        b_state = obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]

        actions.append(action)

        new_obs, _, done, _ = env.step(action)
        obs = new_obs.astype('float32')

        if not done: states.append(obs)

    return states, actions

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