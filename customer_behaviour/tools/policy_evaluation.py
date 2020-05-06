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

###########################
##### Other functions #####
###########################

def get_cluster_labels(T):
    _, indices = np.unique(T, return_index=True)
    indices.sort()
    temp = np.zeros(T.shape)
    for i, j in enumerate(indices, start=1):
        temp[T == T[j]] = i
    return temp

#############################
##### Sample generators #####
#############################

def get_env_and_model(args, model_dir_path, sample_length, model_path=None, only_env=False, n_experts_in_adam_basket=None):
    '''
    Creates and returns
        - the environment specified by the the parameters in args
        - the model which is stored in model_dir_path
        - the observation normalizer stored in model_dir_path.
    '''
    
    if model_path is None: model_path = join(model_dir_path, 'model.npz')

    N = args['n_experts'] if n_experts_in_adam_basket is None else n_experts_in_adam_basket

    # Create environment
    env = gym.make('discrete-buying-events-v0')
    try:
        env.initialize_environment(
            case=args['state_rep'], 
            n_historical_events=args['n_historical_events'], 
            episode_length=sample_length,  # length of agent sample
            n_experts=N,
            n_demos_per_expert=1,
            n_expert_time_steps=sample_length,  # length of expert sample
            seed_agent=args['seed_agent'],
            seed_expert=args['seed_expert'],
            adam_days=args['adam_days']
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

    if only_env: return env

    # Initialize model and observation normalizer
    if 'G_layers' in args:
        hidden_sizes = args['G_layers']
    else:
        hidden_sizes = (64, 64)

    if args['state_rep'] == 21:
        model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 11:
        model = A3CFFSoftmax(2 + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(2 + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 22 or args['state_rep'] == 221:
        model = A3CFFSoftmax(args['n_experts'] + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_experts'] + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 23:
        model = A3CFFSoftmax(args['n_experts'] + args['n_historical_events'], 11, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_experts'] + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 24:
        model = A3CFFSoftmax(10 + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(10 + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 31:
        model = A3CFFSoftmax(10 + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(10 + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 4:
        model = A3CFFSoftmax(args['n_experts'] + 2*args['n_historical_events'], 4, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_experts'] + 2*args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 221:
        model = A3CFFSoftmax(args['n_experts'] + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_experts'] + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 222:
        model = A3CFFSoftmax(args['n_experts'] + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_experts'] + args['n_historical_events'], clip_threshold=5)
    elif args['state_rep'] == 7:
        model = A3CFFSoftmax(2 + args['adam_days'] + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(2 + args['adam_days'] + args['n_historical_events'], 2, clip_threshold=5)
    elif args['state_rep'] == 71:
        model = A3CFFSoftmax(args['adam_days'] + args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['adam_days'] + args['n_historical_events'], 2, clip_threshold=5)
    elif args['state_rep'] == 17:
        model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=hidden_sizes)
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], 2, clip_threshold=5)
    else:
        raise NotImplementedError
    
    # Load model and observation normalizer
    chainer.serializers.load_npz(model_path, model)
    try:
        if args['normalize_obs']:
            chainer.serializers.load_npz(join(model_dir_path, 'obs_normalizer.npz'), obs_normalizer)
        else:
            obs_normalizer = None
    except KeyError:
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
        
        if obs_normalizer:
            b_state = obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]

        actions.append(action)

        new_obs, _, done, _ = env.step(action)
        obs = new_obs.astype('float32')

        if not done: states.append(obs)

    return states, actions



def sample_from_policy2(env, model, obs_normalizer, initial_state=None):
    # This function should only be used in results.py

    xp = np
    phi = lambda x: x

    states = []
    actions = []

    if initial_state is not None:
        env.state = initial_state
        obs = np.array(initial_state).astype('float32')
        env.n_time_steps = 0
    else:
        obs = env.reset().astype('float32')  # Initial state

    states.append(obs)
    done = False
    while not done:
        b_state = batch_states([obs], xp, phi)
        
        if obs_normalizer:
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

def get_val_states(states, actions, n, m):
    val_states = []
    states = np.reshape(states, (-1, states.shape[-1]))
    actions = np.reshape(actions, (-1,))
    
    for s in states:
        temp = [1 if x > 0 else 0 for x in s[-n:]]
        val_states.append(temp)

    val_states = reduce_dimensionality(val_states, m)

    assert len(val_states) == len(actions)

    temp = []
    for vs, a in zip(val_states, actions):
        if np.sum(vs) == len(vs):
            temp.append(vs + [1])
        elif np.sum(vs) + a > m:
            temp.append((n + 1) * [1])
        else:
            temp.append(vs + [a])
    val_states = temp

    return val_states 

def get_cond_val_states(states, actions, n):
    n_trajectories = len(states)

    purchase = []
    no_purchase = []

    for i in range(n_trajectories):
        for temp_state, temp_action in zip(states[i], actions[i]):
            # Extract the n last days
            n_last_days = temp_state[-n:]
            val_state = [1 if x > 0 else 0 for x in n_last_days]
            if temp_action > 0:
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
    if not val_states: return []

    indices = np.argwhere(np.sum(val_states, axis=1) > max_n_purchases)  # list of lists
    indices = [x[0] for x in indices]

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

def extract_time_series(states, n_last_days, n_dummies=10):
    n_trajectories = len(states)

    extracted = []

    for i in range(n_trajectories):
        temp = []
        for s in states[i]:
            history = s[:-n_dummies]

            new_s = []
            for x in history:
                while x > 1:
                    new_s.append(0)
                    x -= 1
                new_s.append(1)
            new_s.reverse()

            temp.append(new_s[-n_last_days:])

        extracted.append(temp)

    return extracted

def get_possible_val_states_2(n_last_days, max_n_purchases):
    possible_val_states = get_possible_val_states(n_last_days, max_n_purchases)

    temp1 = []
    temp2 = []
    for s in possible_val_states[:-1]:
        temp1.append(s + [0])
        if np.sum(s) + 1 > max_n_purchases:
            pass
        else:
            temp2.append(s + [1])
    temp3 = [possible_val_states[-1] + [1]]

    possible_val_states = temp1 + temp2 + temp3

    possible_val_states = sort_possible_val_states(possible_val_states)

    return possible_val_states


def get_distrib(states, actions):
    n_last_days = 7
    max_n_purchases = 2

    val_states = get_val_states(states, actions, n_last_days, max_n_purchases)

    possible_val_states = get_possible_val_states_2(n_last_days, max_n_purchases)

    distrib = get_counts(val_states, possible_val_states, normalize=True)

    return distrib

def get_cond_distribs(states, actions, n_last_days, max_n_purchases, normalize, case):
    if case == 31: states = extract_time_series(states, n_last_days, n_dummies=10)

    purchase, no_purchase = get_cond_val_states(states, actions, n_last_days)

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

def get_mean_purchase_histo(actions):
    n_samples = len(actions)
    hist = np.zeros(10)
    for sample in range(n_samples):
        hist += np.histogram(actions[sample], bins=range(11))[0]
    return hist/n_samples

##############################
##### Plot distributions #####
##############################

def get_info(args):
    info = ''

    for a in ['algo', 'n_experts', 'length_expert_TS', 'episode_length', 'n_training_episodes', 'n_historical_events', \
    'state_rep', 'D_layers', 'G_layers', 'PAC_k', 'gamma', 'noise', 'batchsize', 'n_processes', 'show_D_dummy', 'normalize_obs']:
        try:
            if a == 'state_rep':
                info += a + ': ' + str(args[a]) + '\n'
            elif a == 'normalize_obs':
                info += a + ': ' + str(args[a])
            else:
                info += a + ': ' + str(args[a]) + ' | '
        except KeyError:
            pass

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
