import os
import fnmatch
import shutil
import datetime
import numpy as np
import chainer
import itertools
import chainerrl
from chainerrl.misc.batch_states import batch_states

class Expert():
    def __init__(self, purchases, no_purchases, avg_purchase, avg_no_purchase, purchase_ratio=None):
        self.purchases = purchases
        self.no_purchases = no_purchases

        self.avg_purchase = avg_purchase
        self.avg_no_purchase = avg_no_purchase

        self.purchase_ratio = purchase_ratio

        self.calc_avg_dist_from_centroid()

    def calc_avg_dist_from_centroid(self):
        temp = []
        for d in self.purchases:
            temp.append(pe.get_wd(d, self.avg_purchase, normalize))
        self.avg_dist_purchase = np.mean(temp)

        temp = []
        for d in self.no_purchases:
            temp.append(pe.get_wd(d, self.avg_no_purchase, normalize))
        self.avg_dist_no_purchase = np.mean(temp)

    def calculate_pairwise_distances(self):
        self.distances_purchase = []
        for u, v in itertools.combinations(self.purchases, 2):
            wd = pe.get_wd(u, v, normalize)
            self.distances_purchase.append(wd)

        self.distances_no_purchase = []
        for u, v in itertools.combinations(self.no_purchases, 2):
            wd = pe.get_wd(u, v, normalize)
            self.distances_no_purchase.append(wd)



def get_outdir(algo, case, n_experts, state_rep):

    if not os.path.exists('results'): os.makedirs('results')

    path = os.path.join('results', str(algo))
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, case)
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, str(n_experts) + '_expert(s)')
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, 'case_' + str(state_rep))
    if not os.path.exists(path): os.makedirs(path)

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, time)

    return path


def get_env(case, n_experts):
    if case == 'discrete_events':
        env = 'discrete-buying-events-v0'
    elif case == 'full_receipt':
        env = ''

    return env

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 't', 'y', '1', 'true'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def move_dir(src: str, dst: str, pattern: str = '*'):
    if not os.path.isdir(dst):
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    for f in fnmatch.filter(os.listdir(src), pattern):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

def read_npz_file(path, get_npz_as_list = False, print_npz = True):
    from numpy import load
    ls = [] if get_npz_as_list else None
    data = load(path, allow_pickle=True)
    lst = data.files
    for item in lst:
        print(data[item]) if print_npz else None
        ls.append(data[item])
    return ls

def convert_imports(dir):
    keyword = 'chainerrl'
    replacement = 'customer_behaviour.chainerrl'
    for subdir, dirs, files in os.walk(dir):
        for file in files:
        #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".py"):
                with open(filepath) as f:
                    newText=f.read().replace(keyword, replacement)

                with open(filepath, 'w') as f:
                    f.write(newText)
                
    print('----- .py files that had ' + str(keyword) + ' in it has now been replaced to ' + str(replacement) + ' -----')

def save_plt_as_eps(obj, path):
    obj.savefig(path, format='eps')

def save_plt_as_png(obj, path):
    obj.set_size_inches((12, 8.5), forward=False)
    obj.savefig(path, format='png', dpi=300)

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
    xp = np
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

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

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
    