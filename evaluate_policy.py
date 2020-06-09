import sys
import os, re, json, random, itertools, time
from os.path import join
tools_path = join(os.getcwd(), 'customer_behaviour/tools')
sys.path.insert(1, tools_path)

import policy_evaluation as pe
from result import Result
from tools import save_plt_as_eps, save_plt_as_png
import gym
import custom_gym
import chainer
import chainerrl
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import wasserstein_distance

dir_path = ''  # Add a path to a directory where data is saved

model_path = None # join(os.getcwd(), dir_path, '6570000_checkpoint', 'model.npz')

sample_length = 10000
normalize = True
n_demos_per_expert = 10
n_last_days = 7
max_n_purchases_per_n_last_days = 2
show_info = True
show_plots = True
save_plots = True
cluster_comparison = False

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

    # purchase_ratio(args, model_dir_path)

    evaluate_policy_at_population_level_multiple_categories(args, model_dir_path, ending_eps, ending_png, info)
    evaluate_policy_at_individual_level_multiple_categories(args, model_dir_path, ending_eps, ending_png, info)

    evaluate_policy_at_population_level(args, model_dir_path, ending_eps, ending_png, info)
    # evaluate_policy_at_individual_level(args, model_dir_path, ending_eps, ending_png, info)
    compare_clusters(args, model_dir_path, ending_eps, ending_png, info)
    # visualize_experts(n_experts=10)

    fig_stats = plot_statistics(dir_path)
    fig_path = os.getcwd() + '/' + dir_path + '/figs'
    save_plt_as_png(fig_stats, fig_path + '/stats.png')

def eval_policy(
    a_dir_path,
    a_sample_length = 10000, 
    a_normalize = True,
    a_n_demos_per_expert = 10,
    a_n_last_days = 7,
    a_max_n_purchases_per_n_last_days = 2,
    a_show_info = True,
    a_show_plots = False,
    a_save_plots = True,
    a_cluster_comparison = False
    ):
    
    global dir_path, sample_length, normalize, n_demos_per_expert, n_last_days, max_n_purchases_per_n_last_days, show_info, show_plots, save_plots, cluster_comparison
    dir_path = a_dir_path
    sample_length = a_sample_length
    normalize = a_normalize
    n_demos_per_expert = a_n_demos_per_expert
    n_last_days = a_n_last_days
    max_n_purchases_per_n_last_days = a_max_n_purchases_per_n_last_days
    show_info = a_show_info
    show_plots = a_show_plots
    save_plots = a_save_plots
    cluster_comparison = a_cluster_comparison

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
    # evaluate_policy_at_individual_level(args, model_dir_path, ending_eps, ending_png, info)
    compare_clusters(args, model_dir_path, ending_eps, ending_png, info)

    fig_stats = plot_statistics(dir_path)
    fig_path = os.getcwd() + '/' + dir_path + '/figs'
    save_plt_as_png(fig_stats, fig_path + '/stats.png')
    if not show_plots: plt.close(fig_stats)

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
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

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

    n_experts = 2 if (args['state_rep'] == 24 or args['state_rep'] == 31) else args['n_experts']
    
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
                normalize,
                case=args['state_rep']
                )
            purchases.append(temp_purchase)
            no_purchases.append(temp_no_purchase)

        avg_purchase, avg_no_purchase, _ = pe.get_cond_distribs(
            states, 
            actions, 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize,
            case=args['state_rep']
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
        normalize,
        case=args['state_rep']
        )

    if cluster_comparison and (args['state_rep'] != 24 or args['state_rep'] != 31):
        # Cluster expert data (purcase)
        X = np.array([e.avg_purchase for e in experts])
        T_purchase = fclusterdata(X, 3, 'maxclust', method='single', metric=lambda u, v: wasserstein_distance(u, v))
        T_purchase = pe.get_cluster_labels(T_purchase)

        # Cluster expert data (purcase)
        X = np.array([e.avg_no_purchase for e in experts])
        T_no_purchase = fclusterdata(X, 3, 'maxclust', method='single', metric=lambda u, v: wasserstein_distance(u, v))
        T_no_purchase = pe.get_cluster_labels(T_no_purchase)

        assert np.array_equal(T_purchase, T_no_purchase)
        cluster_indices = [np.argwhere(T_purchase == i) for i in [1, 2, 3]]

        distances_purchase = []
        distances_no_purchase = []
    
    all_distances_purchase = []
    all_distances_no_purchase = []

    for i in range(n_experts):
        # Sample agent data starting with expert's history
        initial_state = random.choice(expert_states[i])
        agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        agent_purchase, agent_no_purchase, _ = pe.get_cond_distribs(
            [agent_states], 
            [agent_actions], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize,
            case=args['state_rep']
            )

        e = experts[i]

        # Compare distributions (purchase)
        if cluster_comparison and (args['state_rep'] != 24 or args['state_rep'] != 31):
            temp = [e.avg_dist_purchase]
            temp.append(pe.get_wd(e.avg_purchase, agent_purchase, normalize))
            temp.append(pe.get_wd(avg_expert_purchase, agent_purchase, normalize))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_purchase, agent_purchase, normalize) for j in cluster_indices[0]]))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_purchase, agent_purchase, normalize) for j in cluster_indices[1]]))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_purchase, agent_purchase, normalize) for j in cluster_indices[2]]))
            distances_purchase.append(temp)

        temp = [pe.get_wd(e.avg_purchase, agent_purchase, normalize) for e in experts]
        temp.append(pe.get_wd(avg_expert_purchase, agent_purchase, normalize))
        all_distances_purchase.append(temp)

        # Compare distributions (no purchase)
        if cluster_comparison and (args['state_rep'] != 24 or args['state_rep'] != 31):
            temp = [e.avg_dist_no_purchase]
            temp.append(pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize))
            temp.append(pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_no_purchase, agent_no_purchase, normalize) for j in cluster_indices[0]]))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_no_purchase, agent_no_purchase, normalize) for j in cluster_indices[1]]))
            temp.append(np.mean([pe.get_wd(experts[j[0]].avg_no_purchase, agent_no_purchase, normalize) for j in cluster_indices[2]]))
            distances_no_purchase.append(temp)

        temp = [pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize) for e in experts]
        temp.append(pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize))
        all_distances_no_purchase.append(temp)

    if cluster_comparison and (args['state_rep'] != 24 or args['state_rep'] != 31):
        ##### Plot distance to one expert #####
        columns = ['Var. in expert cluster', 'Dist. to expert', 'Dist. to avg. expert', 'Dist. to 1st cluster 1', 'Dist. to 2nd cluster', 'Dist. to 3rd cluster']
        index = ['E{}\n({})'.format(i + 1, int(T_purchase[i])) for i in range(n_experts)]

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
        if not show_plots: plt.close(fig)

    ##### Plot distance to all experts #####
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
    fig.subplots_adjust(bottom=0.25)

    columns = ['Expert {}'.format(i + 1) for i in range(n_experts)]
    columns.append('Avg. expert')
    index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

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
    if not show_plots: plt.close(fig)
    
    if show_plots: plt.show()

    '''
    ##### Plot distance to all experts (no purchase) #####
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    fig.subplots_adjust(left=0.25)

    columns = ['Customer {}'.format(i + 1) for i in range(n_experts)]
    columns.append('Avg. customer')
    index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

    all_distances_no_purchase = pd.DataFrame(all_distances_no_purchase, columns=columns, index=index)
    seaborn.heatmap(all_distances_no_purchase, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': "Earth mover's distance"})
    fig.suptitle('Comparison at individual level')

    plt.show()
    '''

############################
############################

def evaluate_policy_at_individual_level(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

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
        normalize,
        case=args['state_rep']
        )
    n_experts = 2 if (args['state_rep'] == 24 or args['state_rep'] == 31) else args['n_experts']
    avg_expert_shopping_ratio = format(avg_expert_n_shopping_days / (n_experts * sample_length), '.2f')

    all_expert_indices = range(n_experts)
    expert_indices_list = []
    for i in range(0, n_experts, 4):
        try:
            expert_indices_list.append(all_expert_indices[i:i+4])
        except IndexError:
            expert_indices_list.append(all_expert_indices[i:])

    for j, expert_indices in enumerate(expert_indices_list):
        fig1, axes1 = plt.subplots(2, 2, sharex='col')
        fig2, axes2 = plt.subplots(2, 2, sharex='col')
        
        for i, ax1, ax2 in zip(expert_indices, axes1.flat, axes2.flat):
            # Sample agent data starting with expert's history
            initial_state = random.choice(expert_states[i])

            agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

            agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
                [agent_states], 
                [agent_actions], 
                n_last_days, 
                max_n_purchases_per_n_last_days, 
                normalize,
                case=args['state_rep']
                )
            agent_shopping_ratio = format(agent_n_shopping_days / sample_length, '.3f')

            expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
                [expert_states[i]], 
                [expert_actions[i]], 
                n_last_days, 
                max_n_purchases_per_n_last_days, 
                normalize,
                case=args['state_rep']
                )
            expert_shopping_ratio = format(expert_n_shopping_days / sample_length, '.3f')

            if args['state_rep'] == 23:
                expert_histo, _ = np.histogram(expert_actions[i], bins=range(11))
                agent_histo, _ = np.histogram(agent_actions, bins=range(11))

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
            ax1.set_title('Expert {}: {}, age {}\nEMD (expert): {:.5f} | EMD (avg. expert): {:.5f}'.format(i+1, sex[i], age[i], wd_purchase, wd_purchase_avg))

            # Plot (no purchase)
            data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase, avg_expert_str: avg_expert_no_purchase}
            pe.bar_plot(ax2, data, colors=None, total_width=0.7)
            ax2.set_title('Expert {}: {}, age {}\nEMD (expert): {:.5f} | EMD (avg. expert): {:.5f}'.format(i+1, sex[i], age[i], wd_purchase, wd_purchase_avg))

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

        if save_plots: save_plt_as_png(fig1, path=join(dir_path, 'figs', 'ind_purchase_' + str(j+1) + ending_png))
        if save_plots: save_plt_as_png(fig2, path=join(dir_path, 'figs', 'ind_no_purchase_' + str(j+1) + ending_png))

        if not show_plots: 
            plt.close(fig1)
            plt.close(fig2)

        if show_plots: plt.show()

############################
############################

def evaluate_policy_at_population_level(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample expert data
    expert_trajectories = env.generate_expert_trajectories(out_dir=None)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']

    expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    n_experts = 2 if (args['state_rep'] == 24 or args['state_rep'] == 31) else args['n_experts']

    # Sample agent data
    agent_states = []
    agent_actions = []
    for i in range(n_experts):
        initial_state = random.choice(expert_states[i])
        # What happens if the agent is repeatedly initialized with history from expert with unique behavior?
        # Should we collect more than n_experts samples (especially if n_experts <= 10)?
        temp_states, temp_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
        agent_states.append(temp_states)
        agent_actions.append(temp_actions)

    agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
        agent_states, 
        agent_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    # Calculate Wasserstein distances
    wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
    wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)
    
    agent_shopping_ratio = format(agent_n_shopping_days / (n_experts * sample_length), '.3f')
    expert_shopping_ratio = format(expert_n_shopping_days / (n_experts * sample_length), '.3f')
    expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
    agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Comparison at population level')

    # Plot (purchase)
    data = {expert_str: expert_purchase, agent_str: agent_purchase}

    pe.bar_plot(ax1, data, colors=None, total_width=0.7)
    ax1.set_xticks([], [])
    ax1.set_title('Purchase | EMD: {:.5f}'.format(wd_purchase))
    # ax1.set_title('Last week | Purchase today')
    ax1.set_ylabel('Probability')

    # Plot (no purchase)
    data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase}
    pe.bar_plot(ax2, data, colors=None, total_width=0.7)
    ax2.set_xticks([], [])
    ax2.set_title('No purchase | EMD: {:.5f}'.format(wd_no_purchase))
    # ax2.set_title('Last week | No purchase today')
    ax2.set_ylabel('Probability')
    
    if show_info: fig.text(0.5, 0.025, info, ha='center')
    if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop' + ending_png))
    if not show_plots: plt.close(fig)

    if args['state_rep'] == 23:
        # Plot histogram of purchase amounts
        expert_amounts = np.ravel(expert_actions)[np.flatnonzero(expert_actions)]
        agent_amounts = np.ravel(agent_actions)[np.flatnonzero(agent_actions)]

        fig, ax = plt.subplots()
        ax.hist(expert_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Expert')
        ax.hist(agent_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Agent')
        ax.set_xlabel('Purchase amount')
        ax.set_ylabel('Normalized frequency')
        ax.legend()

        if show_info: fig.text(0.5, 0.025, info, ha='center')
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop_amounts' + ending_png))
        if not show_plots: plt.close(fig)

    if show_plots: plt.show()

############################
############################

def convert_action(action):
    if action == 0:
        a1 = 1
        a2 = 1
    elif action == 1:
        a1 = 1
        a2 = 0
    elif action == 2:
        a1 = 0
        a2 = 1
    else:
        a1 = 0
        a2 = 0
    return a1, a2

def plot_distr_categories(
        args, 
        info, 
        ending_png, 
        expert_actions, 
        agent_actions, 
        expert_purchase, 
        agent_purchase, 
        expert_no_purchase, 
        agent_no_purchase, 
        agent_n_shopping_days, 
        expert_n_shopping_days, 
        category=1
        ):
    # Calculate Wasserstein distances
    wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
    wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)
    
    n_experts = 2 if (args['state_rep'] == 24 or args['state_rep'] == 31) else args['n_experts']
    agent_shopping_ratio = format(agent_n_shopping_days / (args['n_experts'] * sample_length), '.3f')
    expert_shopping_ratio = format(expert_n_shopping_days / (n_experts * sample_length), '.3f')
    expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
    agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Comparison at population level for category ' + str(category))

    # Plot (purchase)
    data = {expert_str: expert_purchase, agent_str: agent_purchase}

    pe.bar_plot(ax1, data, colors=None, total_width=0.7)
    ax1.set_xticks([], [])
    ax1.set_title('Purchase | EMD: {:.5f}'.format(wd_purchase))
    # ax1.set_title('Last week | Purchase today')
    ax1.set_ylabel('Probability')

    # Plot (no purchase)
    data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase}
    pe.bar_plot(ax2, data, colors=None, total_width=0.7)
    ax2.set_xticks([], [])
    ax2.set_title('No purchase | EMD: {:.5f}'.format(wd_no_purchase))
    # ax2.set_title('Last week | No purchase today')
    ax2.set_ylabel('Probability')
    
    if show_info: fig.text(0.5, 0.025, info, ha='center')
    if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop' + 'cat' + str(category) + ending_png))

    if args['state_rep'] == 23:
        # Plot histogram of purchase amounts
        expert_amounts = np.ravel(expert_actions)[np.flatnonzero(expert_actions)]
        agent_amounts = np.ravel(agent_actions)[np.flatnonzero(agent_actions)]

        fig, ax = plt.subplots()
        ax.hist(expert_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Expert')
        ax.hist(agent_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Agent')
        ax.set_xlabel('Purchase amount')
        ax.set_ylabel('Normalized frequency')
        ax.legend()

        if show_info: fig.text(0.5, 0.025, info, ha='center')
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop_amounts' + ending_png))

def evaluate_policy_at_population_level_multiple_categories(args, model_dir_path, ending_eps, ending_png, info):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample agent data from both categories
    agent_states_1 = []
    agent_states_2 = []
    agent_actions_1 = []
    agent_actions_2 = []
    #for i in range(args['n_experts']):
    for i in range(args['n_experts']):
        # What happens if the agent is repeatedly initialized with history from expert with unique behavior?
        # Should we collect more than n_experts samples (especially if n_experts <= 10)?
        temp_s_1 = []
        temp_s_2 = []
        temp_a_1 = []
        temp_a_2 = []
        temp_states, temp_actions = pe.sample_from_policy(env, model, obs_normalizer)
        for state, action in zip(temp_states, temp_actions):
            if len(state) == args['n_experts'] + 2*args['n_historical_events']: # if dummy variables == n_experts
                s_1, s_2 = np.split(state[args['n_experts']:], 2)
                #state_2 = state[args['n_historical_events']:]                

            elif len(state) == 10 + 2*args['n_historical_events']: # if dummy variables == 10
                s_1, s_2 = np.split(state[10:], 2)
            temp_s_1.append(s_1)
            temp_s_2.append(s_2)
            a1, a2 = convert_action(action)
            temp_a_1.append(a1)
            temp_a_2.append(a2)

        agent_states_1.append(temp_s_1)
        agent_states_2.append(temp_s_2)
        agent_actions_1.append(temp_a_1)
        agent_actions_2.append(temp_a_2)

    agent_purchase_1, agent_no_purchase_1, agent_n_shopping_days_1 = pe.get_cond_distribs(
        agent_states_1, 
        agent_actions_1, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )
    agent_purchase_2, agent_no_purchase_2, agent_n_shopping_days_2 = pe.get_cond_distribs(
        agent_states_1, 
        agent_actions_1, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    # Sample expert data
    expert_trajectories = env.generate_expert_trajectories(out_dir=None)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']

    expert_states_1 = []
    expert_states_2 = []
    expert_actions_1 = []
    expert_actions_2 = []

    for i in range(args['n_experts']):
        # What happens if the agent is repeatedly initialized with history from expert with unique behavior?
        # Should we collect more than n_experts samples (especially if n_experts <= 10)?
        temp_s_1 = []
        temp_s_2 = []
        temp_a_1 = []
        temp_a_2 = []
        temp_states = expert_states[i]
        temp_actions = expert_actions[i]
        for state, action in zip(temp_states, temp_actions):
            if len(state) == args['n_experts'] + 2*args['n_historical_events']: # if dummy variables == n_experts
                s_1, s_2 = np.split(state[args['n_experts']:], 2)               

            elif len(state) == 10 + 2*args['n_historical_events']: # if dummy variables == 10
                s_1, s_2 = np.split(state[10:], 2)
            temp_s_1.append(s_1)
            temp_s_2.append(s_2)
            a1, a2 = convert_action(action)
            temp_a_1.append(a1)
            temp_a_2.append(a2)

        expert_states_1.append(temp_s_1)
        expert_states_2.append(temp_s_2)
        expert_actions_1.append(temp_a_1)
        expert_actions_2.append(temp_a_2)

    expert_purchase_1, expert_no_purchase_1, expert_n_shopping_days_1 = pe.get_cond_distribs(
        expert_states_1, 
        expert_actions_1, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    expert_purchase_2, expert_no_purchase_2, expert_n_shopping_days_2 = pe.get_cond_distribs(
        expert_states_2, 
        expert_actions_2, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    plot_distr_categories(
        args, 
        info, 
        ending_png, 
        expert_actions_1, 
        agent_actions_1, 
        expert_purchase_1, 
        agent_purchase_1, 
        expert_no_purchase_1, 
        agent_no_purchase_1, 
        agent_n_shopping_days_1, 
        expert_n_shopping_days_1, 
        category=1
        )
    plot_distr_categories(args, 
        info, 
        ending_png, 
        expert_actions_2, 
        agent_actions_2, 
        expert_purchase_2,
        agent_purchase_2, 
        expert_no_purchase_2, 
        agent_no_purchase_2, 
        agent_n_shopping_days_2, 
        expert_n_shopping_days_2, 
        category=2
        )
    plt.show()

############################
############################

def plot_statistics(dir_path):
    discriminator_loss, policy_loss, average_rewards, average_D_output, average_mod_rewards, value_loss, value, n_updates, episodes, average_entropy = read_scores_txt(dir_path)

    fig = plt.figure()
    plt.subplot(2,4,1)
    plt.plot(episodes, discriminator_loss)
    plt.xlabel('Episode')
    plt.ylabel('Average discriminator loss')

    plt.subplot(2,4,2)
    plt.plot(episodes, policy_loss)
    plt.xlabel('Episode')
    plt.ylabel('Average policy loss')

    plt.subplot(2,4,3)
    plt.plot(episodes, value)
    plt.xlabel('Episode')
    plt.ylabel('Average value')

    plt.subplot(2,4,4)
    plt.plot(episodes, value_loss)
    plt.xlabel('Episode')
    plt.ylabel('Average value loss')

    plt.subplot(2,4,5)
    plt.plot(episodes, average_entropy)
    plt.xlabel('Episode')
    plt.ylabel('Average entropy')

    plt.subplot(2,4,6)
    plt.plot(episodes, average_D_output)
    plt.xlabel('Episode')
    plt.ylabel('Average D output')

    plt.subplot(2,4,7)
    plt.plot(episodes, average_mod_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average mod reward')

    plt.subplot(2,4,8)
    plt.plot(episodes, average_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Average reward')

    plt.tight_layout()
    if show_plots: plt.show()
    
    return fig

def read_scores_txt(dir_path):
    path = os.getcwd() + '/' + dir_path + '/scores.txt'
    file_obj = open(path, 'r') 
    lines = file_obj.readlines()
    
    discriminator_loss = []
    policy_loss = []
    average_rewards = []
    average_D_output = []
    average_mod_rewards = []
    value_loss = []
    value = []
    n_updates = []
    episodes = []
    average_entropy = []
    
    for idx, line in enumerate(lines):
        if idx == 0:
            line1 = line.split(' ')[0]
            columns = re.split(r'\t+', line1)
            columns = [x.rstrip('\n\r') for x in columns]

            i_dl = columns.index('average_discriminator_loss')
            i_pl = columns.index('average_policy_loss')
            i_D = columns.index('average_D_output')
            i_m_r = columns.index('average_mod_rewards')
            i_r = columns.index('average_rewards')
            i_v = columns.index('average_value')
            i_vl = columns.index('average_value_loss')
            i_u = columns.index('n_updates')
            i_e = columns.index('episodes')
            i_ae = columns.index('average_entropy')

        if idx > 0:  # We do not want the column names
            line1 = line.split(' ')[0]
            line2 = re.split(r'\t+', line1)
            
            discriminator_loss.append(float(line2[i_dl].rstrip('\n\r')))
            policy_loss.append(float(line2[i_pl].rstrip('\n\r')))     
            average_rewards.append(float(line2[i_r].rstrip('\n\r')))
            average_D_output.append(float(line2[i_D].rstrip('\n\r')))
            average_mod_rewards.append(float(line2[i_m_r].rstrip('\n\r')))
            value_loss.append(float(line2[i_vl].rstrip('\n\r')))
            value.append(float(line2[i_v].rstrip('\n\r')))
            n_updates.append(float(line2[i_u].rstrip('\n\r')))
            episodes.append(float(line2[i_e].rstrip('\n\r')))
            average_entropy.append(float(line2[i_ae].rstrip('\n\r')))

    file_obj.close()

    return discriminator_loss, policy_loss, average_rewards, average_D_output, average_mod_rewards, value_loss, value, n_updates, episodes, average_entropy

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

    # Cluster expert data (purcase)
    X = np.array([e.avg_purchase for e in experts])
    T_purchase = fclusterdata(X, 3, 'maxclust', method='single', metric=lambda u, v: wasserstein_distance(u, v))

    # Cluster expert data (no purcase)
    X = np.array([e.avg_no_purchase for e in experts])
    T_no_purchase = fclusterdata(X, 3, 'maxclust', method='single', metric=lambda u, v: wasserstein_distance(u, v))

    ##### Plot distributions #####

    # Expert 1–6
    fig1, axes1 = plt.subplots(2, 3, sharex='col')
    fig2, axes2 = plt.subplots(2, 3, sharex='col')

    for i, (ax1, ax2) in enumerate(zip(axes1.flat, axes2.flat)):
        e = experts[i]

        ax1.set_title('Expert {}: {}, age {} | Purchase ratio: {:.3f}'.format(i+1, sex[i], age[i], e.purchase_ratio))
        ax2.set_title('Expert {}: {}, age {} | Purchase ratio: {:.3f}'.format(i+1, sex[i], age[i], e.purchase_ratio))

        data = {'': e.avg_purchase}
        pe.bar_plot(ax1, data, colors=None, total_width=0.7, legend=False)

        data = {'': e.avg_no_purchase}
        pe.bar_plot(ax2, data, colors=None, total_width=0.7, legend=False)

    fig1.suptitle('Purchase')
    fig2.suptitle('No purchase')

    # Expert 7-10
    fig1, axes1 = plt.subplots(2, 2, sharex='col')
    fig2, axes2 = plt.subplots(2, 2, sharex='col')

    for i, (ax1, ax2) in enumerate(zip(axes1.flat, axes2.flat), start=6):
        e = experts[i]

        ax1.set_title('Expert {}: {}, age {} | Purchase ratio: {:.3f}'.format(i+1, sex[i], age[i], e.purchase_ratio))
        ax2.set_title('Expert {}: {}, age {} | Purchase ratio: {:.3f}'.format(i+1, sex[i], age[i], e.purchase_ratio))

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

def get_pr(sequence, n=None):
    s = sequence if n is None else sequence[:n]
    pr = np.count_nonzero(s) / n
    return pr

def purchase_ratio(args, model_dir_path):
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length=10000, model_path=model_path)

    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    sex = ['F' if s == 1 else 'M' for s in expert_trajectories['sex']]
    age = [int(a) for a in expert_trajectories['age']]

    for i, (e_states, e_actions) in enumerate(zip(expert_states, expert_actions)):  # Loop over experts
        print('Expert %d' % (i+1))

        # Sample data from agent
        initial_state = e_states[0]  # random.choice(e_states)
        _, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        temp1 = []
        temp2 = []

        for n in [100, 500, 1000, 5000, 10000]:
            e_pr = get_pr(e_actions, n)
            a_pr = get_pr(agent_actions, n)

            temp1.append(e_pr)
            temp2.append(a_pr)

        fig, ax = plt.subplots()
        ax.plot([100, 500, 1000, 5000, 10000], temp1, label='Expert')
        ax.plot([100, 500, 1000, 5000, 10000], temp2, label='Agent')
        ax.legend()

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
