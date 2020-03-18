import os
import gym
import custom_gym
import chainer
import chainerrl
import json
import itertools
import seaborn
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import customer_behaviour.tools.policy_evaluation as pe
from os.path import join
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from customer_behaviour.tools.tools import save_plt_as_eps, save_plt_as_png

# dir_path = 'results_anton/2020-03-10_14-04-29'  # 10 experts | 96 historical events  | length_expert_TS = 256  | 10 000 episodes
dir_path = 'results_anton/2020-03-12_11-14-55'  # 10 experts | 256 historical events | length_expert_TS = 256  | 10 000 episodes
# dir_path = 'results_anton/2020-03-16_20-13-19'  # 10 experts | 256 historical events | length_expert_TS = 1024 | 15 000 episodes

sample_length = 10000
normalize = True
n_demos_per_expert = 10
n_last_days = 7
max_n_purchases_per_n_last_days = 2
show_info = True
save_plots = True

def main():
    # Load arguments
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'

    evaluate_policy_at_population_level(args, model_dir_path, ending_eps, ending_png, info)
    evaluate_policy_at_individual_level(args, model_dir_path, ending_eps, ending_png, info)
    compare_clusters(args, model_dir_path, ending_eps, ending_png, info)
    # visualize_experts(n_experts=10)

############################
############################

class Expert():
    def __init__(self, purchases, no_purchases, avg_purchase, avg_no_purchase, purchase_ratio=None):
        self.purchases = purchases
        self.no_purchases = no_purchases

        self.avg_purchase = avg_purchase
        self.avg_no_purchase = avg_no_purchase

        self.purchase_ratio = purchase_ratio

        self.calc_avg_dist_from_centroid()
        # self.calculate_pairwise_distances()

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

def compare_clusters(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Get multiple samples from each expert
    assert (sample_length % n_demos_per_expert) == 0
    expert_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=n_demos_per_expert, 
        n_expert_time_steps=int(sample_length / n_demos_per_expert)
        )
    expert_states = np.array(expert_trajectories['states'])
    expert_actions = np.array(expert_trajectories['actions'])
    sex = ['F' if s == 1 else 'M' for s in expert_trajectories['sex']]
    age = [int(a) for a in expert_trajectories['age']]

    n_experts = args['n_experts']

    experts = []

    for states, actions in zip(np.split(expert_states, n_experts), np.split(expert_actions, n_experts)):  # Loop over experts
        purchases = []
        no_purchases = []

        for s, a in zip(states, actions):  # Loop over demonstrations       
            temp_purchase, temp_no_purchase, _ = pe.get_cond_distribs(
                [s], 
                [a], 
                n_last_days, 
                max_n_purchases_per_n_last_days, 
                normalize
                )
            purchases.append(temp_purchase)
            no_purchases.append(temp_no_purchase)

        avg_purchase, avg_no_purchase, _ = pe.get_cond_distribs(
            states, 
            actions, 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        experts.append(Expert(purchases, no_purchases, avg_purchase, avg_no_purchase))

    # Calculate average expert behavior
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']

    avg_expert_purchase, avg_expert_no_purchase, _ = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )

    distances_purchase = []
    distances_no_purchase = []
    all_distances_purchase = []
    all_distances_no_purchase = []

    for i in range(n_experts):
        # Sample agent data starting with expert's history
        initial_state = expert_states[i][0].copy()
        agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        agent_purchase, agent_no_purchase, _ = pe.get_cond_distribs(
            [agent_states], 
            [agent_actions], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        e = experts[i]

        # Compare distributions (purchase)
        temp = [e.avg_dist_purchase]
        temp.append(pe.get_wd(e.avg_purchase, agent_purchase, normalize))
        temp.append(pe.get_wd(avg_expert_purchase, agent_purchase, normalize))
        distances_purchase.append(temp)

        temp = [pe.get_wd(e.avg_purchase, agent_purchase, normalize) for e in experts]
        temp.append(pe.get_wd(avg_expert_purchase, agent_purchase, normalize))
        all_distances_purchase.append(temp)

        # Compare distributions (no purchase)
        temp = [e.avg_dist_no_purchase]
        temp.append(pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize))
        temp.append(pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize))
        distances_no_purchase.append(temp)

        temp = [pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize) for e in experts]
        temp.append(pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize))
        all_distances_no_purchase.append(temp)

    ##### Plot distance to one expert #####
    columns = ['Var. in expert cluster', 'Dist. to expert', 'Dist. to avg. expert']
    index = ['E{}'.format(i + 1) for i in range(n_experts)]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.30)

    # Plot distance (purchase)
    distances_purchase = pd.DataFrame(distances_purchase, columns=columns, index=index)
    seaborn.heatmap(distances_purchase, cmap='BuPu', ax=ax1, linewidth=1, cbar_kws={'label': 'EMD'})
    ax1.set_title('Purchase')

    distances_no_purchase = pd.DataFrame(distances_no_purchase, columns=columns, index=index)
    seaborn.heatmap(distances_no_purchase, cmap='BuPu', ax=ax2, linewidth=1, cbar_kws={'label': 'EMD'})
    ax2.set_title('No purchase')

    if show_info: fig.text(0.5, 0.025, info, ha='center')
    if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'heatmap' + ending_png))

    ##### Plot distance to all experts #####

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
    fig.subplots_adjust(bottom=0.25)

    columns = ['E{}'.format(i + 1) for i in range(n_experts)]
    columns.append('Avg. expert')
    index = ['E{}'.format(i + 1) for i in range(n_experts)]

    # Plot the distance between each expert cluster (purcahse)
    all_distances_purchase = pd.DataFrame(all_distances_purchase, columns=columns, index=index)
    seaborn.heatmap(all_distances_purchase, cmap='BuPu', ax=ax1, linewidth=1, cbar_kws={'label': 'EMD'})
    ax1.set_title('Purchase')

    # Plot the distance between each expert cluster (no purcahse)
    all_distances_no_purchase = pd.DataFrame(all_distances_no_purchase, columns=columns, index=index)
    seaborn.heatmap(all_distances_no_purchase, cmap='BuPu', ax=ax2, linewidth=1, cbar_kws={'label': 'EMD'})
    ax2.set_title('No purchase')

    if show_info: fig.text(0.5, 0.025, info, ha='center')
    if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'heatmap_all' + ending_png))

    plt.show()

############################
############################

def visualize_experts(n_experts=10):
    env = gym.make('discrete-buying-events-v0')
    env.initialize_environment(
        case=21, 
        n_historical_events=96, 
        episode_length=sample_length,  # length of agent sample
        n_experts=n_experts,
        n_demos_per_expert=1,
        n_expert_time_steps=sample_length  # length of expert sample
        )

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Get multiple samples from each expert
    assert (sample_length % n_demos_per_expert) == 0
    expert_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=n_demos_per_expert, 
        n_expert_time_steps=int(sample_length / n_demos_per_expert)
        )
    expert_states = np.array(expert_trajectories['states'])
    expert_actions = np.array(expert_trajectories['actions'])
    sex = ['F' if s == 1 else 'M' for s in expert_trajectories['sex']]
    age = [int(a) for a in expert_trajectories['age']]

    experts = []

    for states, actions in zip(np.split(expert_states, n_experts), np.split(expert_actions, n_experts)):  # Loop over experts
        purchases = []
        no_purchases = []

        for s, a in zip(states, actions):  # Loop over demonstrations       
            temp_purchase, temp_no_purchase, _ = pe.get_cond_distribs(
                [s], 
                [a], 
                n_last_days, 
                max_n_purchases_per_n_last_days, 
                normalize
                )
            purchases.append(temp_purchase)
            no_purchases.append(temp_no_purchase)

        avg_purchase, avg_no_purchase, n_shopping_days = pe.get_cond_distribs(
            states, 
            actions, 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        purchase_ratio = n_shopping_days / sample_length

        experts.append(Expert(purchases, no_purchases, avg_purchase, avg_no_purchase, purchase_ratio))

    # Calculate average expert behavior
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']

    avg_expert_purchase, avg_expert_no_purchase, _ = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )

    ##### Plot distributions #####

    fig1, axes1 = plt.subplots(2, 3, sharex='col')
    fig2, axes2 = plt.subplots(2, 3, sharex='col')

    for i, (ax1, ax2) in enumerate(zip(axes1.flat, axes2.flat)):
        e = experts[i]

        ax1.set_title('{} | Age {} | Purchase ratio: {:.3f}'.format(sex[i], age[i], e.purchase_ratio))
        ax2.set_title('{} | Age {} | Purchase ratio: {:.3f}'.format(sex[i], age[i], e.purchase_ratio))

        data = {'': e.avg_purchase}
        pe.bar_plot(ax1, data, colors=None, total_width=0.7, legend=False)

        data = {'': e.avg_no_purchase}
        pe.bar_plot(ax2, data, colors=None, total_width=0.7, legend=False)

    fig1.suptitle('Purchase')
    fig2.suptitle('No purchase')

    ##### Look at distances between the experts #####

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
    fig.subplots_adjust(bottom=0.25)

    columns = ['E{}'.format(i + 1) for i in range(n_experts)]
    columns.append('Avg. expert')
    index = ['E{} ({}, {} y)'.format(i + 1, sex[i], age[i]) for i in range(n_experts)]
    index.append('Avg. expert')

    # Plot the distance between each expert cluster (purcahse)
    temp = [e.avg_purchase for e in experts]
    temp.append(avg_expert_purchase)
    df_purchase = pd.DataFrame(squareform(pdist(np.array(temp), lambda u, v: wasserstein_distance(u, v))),
        columns=columns,
        index=index
        )
    seaborn.heatmap(df_purchase, cmap='OrRd', ax=ax1, linewidth=1, cbar_kws={'label': 'EMD'})
    ax1.set_title('Purchase')

    # Plot the distance between each expert cluster (no purcahse)
    temp = [e.avg_no_purchase for e in experts]
    temp.append(avg_expert_no_purchase)
    df_no_purchase = pd.DataFrame(squareform(pdist(np.array(temp), lambda u, v: wasserstein_distance(u, v))),
        columns=columns,
        index=index
        )
    seaborn.heatmap(df_no_purchase, cmap='OrRd', ax=ax2, linewidth=1, cbar_kws={'label': 'EMD'})
    ax2.set_title('No purchase')

    plt.show()

############################
############################

def evaluate_policy_at_individual_level(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample expert data to calculate average expert behavior
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    sex = ['F' if s == 1 else 'M' for s in expert_trajectories['sex']]
    age = [int(a) for a in expert_trajectories['age']]

    avg_expert_purchase, avg_expert_no_purchase, avg_expert_n_shopping_days = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )
    n_experts = args['n_experts']
    avg_expert_shopping_ratio = format(avg_expert_n_shopping_days / (sample_length *  n_experts), '.2f')

    fig1, axes1 = plt.subplots(2, 2, sharex='col')
    fig2, axes2 = plt.subplots(2, 2, sharex='col')

    expert_indices = np.random.choice(range(n_experts), 4, replace=False)  # Choose 4 experts

    for i, ax1, ax2 in zip(expert_indices, axes1.flat, axes2.flat):
        # Sample agent data starting with expert's history
        initial_state = expert_states[i][0].copy()
        agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
            [agent_states], 
            [agent_actions], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )
        agent_shopping_ratio = format(agent_n_shopping_days / sample_length, '.3f')

        expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
            [expert_states[i]], 
            [expert_actions[i]], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )
        expert_shopping_ratio = format(expert_n_shopping_days / sample_length, '.3f')

        # Calculate Wasserstein distances
        wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
        wd_purchase_avg = pe.get_wd(avg_expert_purchase, agent_purchase, normalize)
        wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)
        wd_no_purchase_avg = pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize)

        expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
        agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'
        avg_expert_str = 'Avg. expert (p.r.: ' + str(avg_expert_shopping_ratio) + ')'

        # Plot (purchase)
        data = {expert_str: expert_purchase, agent_str: agent_purchase, avg_expert_str: avg_expert_purchase}
        pe.bar_plot(ax1, data, colors=None, total_width=0.7)
        ax1.set_title('Expert: {}, age {}\nEMD (expert): {:.5f} | EMD (avg. expert): {:.5f}'.format(sex[i], age[i], wd_purchase, wd_purchase_avg))

        # Plot (no purchase)
        data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase, avg_expert_str: avg_expert_no_purchase}
        pe.bar_plot(ax2, data, colors=None, total_width=0.7)
        ax2.set_title('Expert: {}, age {}\nEMD (expert): {:.5f} | EMD (avg. expert): {:.5f}'.format(sex[i], age[i], wd_purchase, wd_purchase_avg))

    fig1.suptitle('Comparison at individual level (purchase)')
    fig2.suptitle('Comparison at individual level (no purchase)')

    if show_info:
        for ax1, ax2 in zip(axes1[1][:], axes2[1][:]):
            ax1.set_xticks([], [])
            ax2.set_xticks([], [])
        fig1.text(0.5, 0.025, info, ha='center')
        fig2.text(0.5, 0.025, info, ha='center')
    else:
        fig1.subplots_adjust(bottom=0.2)
        fig2.subplots_adjust(bottom=0.2)
        for ax1, ax2 in zip(axes1[1][:], axes2[1][:]):
            pe.set_xticks(ax1, possible_val_states, max_n_purchases_per_n_last_days)
            pe.set_xticks(ax2, possible_val_states, max_n_purchases_per_n_last_days)

    if save_plots: save_plt_as_png(fig1, path=join(dir_path, 'figs', 'individual_purchase' + ending_png))
    if save_plots: save_plt_as_png(fig2, path=join(dir_path, 'figs', 'individual_no_purchase' + ending_png))

    plt.show()

############################
############################

def evaluate_policy_at_population_level(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample agent data
    agent_states = []
    agent_actions = []
    for i in range(args['n_experts']):
        # What happens if the agent is repeatedly initialized with history from expert with unique behavior?
        # Should we collect more than n_experts samples (especially if n_experts <= 10)?
        temp_states, temp_actions = pe.sample_from_policy(env, model, obs_normalizer)
        agent_states.append(temp_states)
        agent_actions.append(temp_actions)

    agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
        agent_states, 
        agent_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )

    # Sample expert data
    expert_trajectories = env.generate_expert_trajectories(out_dir=None)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']

    expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )

    # Calculate Wasserstein distances
    wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
    wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)
    
    n_sampled_days = sample_length * args['n_experts']
    agent_shopping_ratio = format(agent_n_shopping_days / n_sampled_days, '.3f')
    expert_shopping_ratio = format(expert_n_shopping_days / n_sampled_days, '.3f')
    expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
    agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Comparison at population level')

    # Plot (purchase)
    data = {expert_str: expert_purchase, agent_str: agent_purchase}
    pe.bar_plot(ax1, data, colors=None, total_width=0.7)
    ax1.set_xticks([], [])
    ax1.set_title('Purchase | EMD: {:.5f}'.format(wd_purchase))

    # Plot (no purchase)
    data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase}
    pe.bar_plot(ax2, data, colors=None, total_width=0.7)
    ax2.set_xticks([], [])
    ax2.set_title('No purchase | EMD: {:.5f}'.format(wd_no_purchase))
    
    if show_info: fig.text(0.5, 0.025, info, ha='center')
    if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'population' + ending_png))

    plt.show()

############################
############################

if __name__ == '__main__':
    main()

############################
############################

'''
pca = PCA(n_components=3)

purchase_pca = pca.fit_transform(all_purchase)
purchase_pca = pd.DataFrame(purchase_pca)
purchase_pca.columns = ['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)]

purchase_pca['Expert']=['Expert {}'.format(i) for i in np.repeat(range(1, n_experts + 1), n_demos_per_expert)]

purchase_pca.head()

fig, ax = plt.subplots()
experts = ['Expert {}'.format(i) for i in range(1, n_experts + 1)]
for e in experts:
    indices = purchase_pca['Expert'] == e
    ax.scatter(purchase_pca.loc[indices, 'PC1'], purchase_pca.loc[indices, 'PC2']) 

ax.legend(experts)
ax.grid()

plt.show()
'''

'''
evaluate_policy_at_population_level

# Plot (purchase)
fig, ax = plt.subplots()
data = {expert_str: expert_purchase, agent_str: agent_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.7)
pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
fig.subplots_adjust(bottom=0.3)
fig.suptitle('Purchase')
ax.set_title('EMD: {:.5f}'.format(wd_purchase))
if show_info: fig.text(0.5, 0.025, info, ha='center')
if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'population_purchase' + ending_png))

# Plot (no purchase)
fig, ax = plt.subplots()
data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.7)
pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
fig.subplots_adjust(bottom=0.3)
fig.suptitle('No purchase')
fig.text(0.5, 0.025, info, ha='center')
if show_info: ax.set_title('EMD: {:.5f}'.format(wd_no_purchase))
if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'population_no_purchase' + ending_png))
'''

'''
compare_clusters

##### Plot distance to all experts #####

columns = ['E{}'.format(i + 1) for i in range(n_experts)]
columns.append('Avg. expert')
index = ['E{}'.format(i + 1) for i in range(n_experts)]

fig, ax = plt.subplots()
all_distances_purchase = pd.DataFrame(all_distances_purchase, columns=columns, index=index)
seaborn.heatmap(all_distances_purchase, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'EMD'})
fig.subplots_adjust(bottom=0.35)
fig.suptitle('Purchase')
if show_info: fig.text(0.5, 0.025, info, ha='center')
if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'heatmap_all_purchase' + ending_png))

fig, ax = plt.subplots()
all_distances_no_purchase = pd.DataFrame(all_distances_no_purchase, columns=columns, index=index)
seaborn.heatmap(all_distances_no_purchase, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'EMD'})
fig.subplots_adjust(bottom=0.35)
fig.suptitle('No purchase')
if show_info: fig.text(0.5, 0.025, info, ha='center')
if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'heatmap_all_no_purchase' + ending_png))
'''
