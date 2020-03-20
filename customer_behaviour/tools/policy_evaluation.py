import gym
import custom_gym
import chainer
import chainerrl
import itertools
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats import wasserstein_distance

import sys
sys.path.append('..')

from main import A3CFFSoftmax
from chainerrl.misc.batch_states import batch_states

#############################
##### Sample generators #####
#############################

def get_env_and_model(args, model_dir_path, sample_length):
    '''
    Creates and returns
        - the environment specified by the the parameters in args
        - the model which is stored in model_dir_path
        - the observation normalizer stored in model_dir_path.
    '''
    
    # Create environment
    env = gym.make('discrete-buying-events-v0')
    try:
        env.initialize_environment(
            case=args['state_rep'], 
            n_historical_events=args['n_historical_events'], 
            episode_length=sample_length,  # length of agent sample
            n_experts=args['n_experts'],
            n_demos_per_expert=1,
            n_expert_time_steps=sample_length,  # length of expert sample
            seed_agent=args['seed_agent'],
            seed_expert=args['seed_expert']
            )
    except KeyError:
        # seed_agent was not an argument 
        env.initialize_environment(
            case=args['state_rep'], 
            n_historical_events=args['n_historical_events'], 
            episode_length=sample_length,  # length of agent sample
            n_experts=args['n_experts'],
            n_demos_per_expert=1,
            n_expert_time_steps=sample_length,  # length of expert sample
            seed_agent=True,
            seed_expert=args['seed_expert']
            )

    # Initialize model and observation normalizer
    if args['state_rep'] == 21:
        model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=(64, 64))
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 11:
        model = A3CFFSoftmax(2 + args['n_historical_events'], 2, hidden_sizes=(64, 64))
        obs_normalizer = chainerrl.links.EmpiricalNormalization(2 + args['n_historical_events'], clip_threshold=5)
    else:
        raise NotImplementedError
    
    # Load model and observation normalizer
    chainer.serializers.load_npz(join(model_dir_path, 'model.npz'), model)
    chainer.serializers.load_npz(join(model_dir_path, 'obs_normalizer.npz'), obs_normalizer)

    return env, model, obs_normalizer

def sample_from_policy(env, model, obs_normalizer, initial_state=None):
    xp = np
    phi = lambda x: x

    states = []
    actions = []

    obs = env.reset().astype('float32')  # Initial state

    if initial_state is not None:
        env.state = initial_state
        obs = np.array(initial_state).astype('float32')

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

#############################
##### Validation states #####
#############################

def get_cond_val_states(states, actions, n):
    n_trajectories = len(states)

    purchase = []
    no_purchase = []

    for i in range(n_trajectories):
        for temp_state, temp_action in zip(states[i], actions[i]):
            # Extract the n last days
            n_last_days = temp_state[-n:]
            val_state = [int(x) for x in n_last_days]
            if temp_action == 1:
                purchase.append(val_state)
            else:
                no_purchase.append(val_state)

    return purchase, no_purchase

def get_possible_val_states(n_last_days, max_n_purchases):
    possible_val_states = [list(x) for x in itertools.product([0, 1], repeat=n_last_days)]
    possible_val_states = reduce_dimensionality(possible_val_states, max_n_purchases, True)
    possible_val_states = sort_possible_val_states(possible_val_states)
    return possible_val_states

def reduce_dimensionality(val_states, max_n_purchases, keep_only_unique=False):
    '''
    val_states: [[v_{0}], [v_{1}], ..., [v_{n-2}], [v_{n-1}]]
    max_n_purchases: The maximum number of purchases that is allowed in a validation state
    '''
    if not val_states:
        return []
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

def sort_possible_val_states(possible_val_states):
    temp = possible_val_states.copy()
    temp_splitted = []
    s = 0
    while sum(len(x) for x in temp_splitted) < len(temp):
        indices = np.argwhere(np.sum(temp, axis=1) == s)  # list of lists
        t = [temp[i[0]] for i in indices]
        t.sort(reverse=True)
        temp_splitted.append(t)
        s += 1
    return [item for sublist in temp_splitted for item in sublist]

#####################################
##### Conditional distributions #####
#####################################

def get_cond_distribs(states, actions, n_last_days, max_n_purchases, normalize):
    purchase, no_purchase = get_cond_val_states(states, actions, n_last_days)
    print(len(purchase))
    print(len(no_purchase))

    # Reduce dimensionality by merging all validation states with more than max_n_purchases purchases to a single state
    purchase = reduce_dimensionality(purchase, max_n_purchases)
    no_purchase = reduce_dimensionality(no_purchase, max_n_purchases)

    possible_val_states = get_possible_val_states(n_last_days, max_n_purchases)

    counts_purchase = get_counts(purchase, possible_val_states, normalize)
    counts_no_purchase = get_counts(no_purchase, possible_val_states, normalize)

    return counts_purchase, counts_no_purchase, len(purchase)

def get_counts(observed_val_states, possible_val_states, normalize=False):
    counts = len(possible_val_states) * [0]
    for temp in observed_val_states:
        i = possible_val_states.index(temp)
        counts[i] += 1
    if normalize:
        counts = list(np.array(counts) / np.sum(counts))
    return counts

def get_wd(u, v, uv_normalized):
    if uv_normalized:
        wd = wasserstein_distance(u, v)
    else:
        uu = np.array(u)
        vv = np.array(v)
        wd = wasserstein_distance(uu / np.sum(uu), vv / np.sum(vv))
    return wd

##############################
##### Plot distributions #####
##############################

def get_info(args):
    algorithm = args['algo'].upper()
    n_experts = args['n_experts']
    expert_length = args['length_expert_TS']
    episode_length = args['episode_length']
    n_training_episodes = args['n_training_episodes']
    n_historical_events = args['n_historical_events']

    info = 'Algorithm: ' + algorithm + ' | ' \
         + 'Number of training episodes: ' + str(n_training_episodes) + ' | ' \
         + 'Episode length: ' + str(episode_length) + ' | ' \
         + 'Number of experts: ' + str(n_experts) + ' | ' \
         + 'Length of expert trajectory: ' + str(expert_length) + ' | ' \
         + 'Length of purchase history: ' + str(n_historical_events)

    return info

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

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax: matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x": [1,2,3],
            "y": [1,2,3],
            "z": [1,2,3],
        }

    colors: array-like, optional, default: None
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle.

    total_width: float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80 % of the x-axis is covered
        by bars and 20 % will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
