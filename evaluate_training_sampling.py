import sys
sys.path.insert(1, '/Users/victor/Documents/CAS_2/Master_thesis/CustomerBehaviour/customer_behaviour/tools')
import os
import json
import seaborn
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import policy_evaluation as pe
from os.path import join
from tools import save_plt_as_png
#from evaluate_policy import Expert
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import wasserstein_distance
from matplotlib.ticker import MaxNLocator

#dir_path = 'temp/2020-04-15_12-47-19'
dir_path = 'ozzy_results/temp/2020-04-17_07-41-43'

sample_length = 10000
normalize = True
n_last_days = 7
max_n_purchases_per_n_last_days = 2
save_plots = True
show_plots = True
show_info = True

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

def eval_training(
    a_dir_path=None, 
    a_sample_length=10000, 
    a_normalize=True,
    a_n_last_days=7,
    a_max_n_purchases_per_n_last_days=2,
    a_save_plots=True,
    a_show_plots=False,
    a_show_info=True
    ):
    
    if a_dir_path:
        global dir_path, sample_length, normalize, n_last_days, max_n_purchases_per_n_last_days, save_plots, show_plots, show_info

        dir_path = a_dir_path
        sample_length = a_sample_length
        normalize = a_normalize
        n_last_days = a_n_last_days
        max_n_purchases_per_n_last_days = a_max_n_purchases_per_n_last_days
        save_plots = a_save_plots
        show_plots = a_show_plots
        show_info = a_show_info

    # Load arguments
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)
    if show_info: info = pe.get_info(args)
    ending_png = '_normalize.png' if normalize else '.png'

    # Create environment
    final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)
    env = pe.get_env_and_model(args, final_model_dir_path, sample_length, only_env=True)

    # Sample expert data
    expert_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=1,
        n_expert_time_steps=sample_length
        )
    expert_states = np.array(expert_trajectories['states'])
    expert_actions = np.array(expert_trajectories['actions'])

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

    expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize,
        case=args['state_rep']
        )

    # Load agent data from models
    def get_key_from_path(path):
        temp = path.split('/')[-1]
        steps = int(temp.split('_')[0])  # training steps
        return steps

    model_dir_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
    model_dir_paths.sort(key=get_key_from_path)

    training_purchase = []
    training_no_purcahse = []

    # training_n_clusters_purchase = []
    # training_n_clusters_no_purchase = []

    for mdp in model_dir_paths:
        n_steps = get_key_from_path(mdp)
        
        env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length)

        # Sample from model
        agent_states = []
        agent_actions = []
        for i in range(n_experts):
            initial_state = random.choice(expert_states[i])
            temp_states, temp_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            agent_states.append(temp_states)
            agent_actions.append(temp_actions)

        ##### Comparison at population level #####

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

        training_purchase.append(wd_purchase)
        training_no_purcahse.append(wd_no_purchase)
        
        n_sampled_days = sample_length * args['n_experts']
        agent_shopping_ratio = format(agent_n_shopping_days / n_sampled_days, '.3f')
        expert_shopping_ratio = format(expert_n_shopping_days / n_sampled_days, '.3f')
        expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
        agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Number of traning steps: %d' % n_steps)

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
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop_' + str(n_steps) + ending_png))

        plt.close(fig)

        ##### Comparison at individual level #####

        all_distances_no_purchase = []
        all_agent_purchase = []
        all_agent_no_purchase = []
        
        for i in range(n_experts):
            agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
                [agent_states[i]], 
                [agent_actions[i]], 
                n_last_days, 
                max_n_purchases_per_n_last_days, 
                normalize,
                case=args['state_rep']
                )

            all_agent_purchase.append(agent_purchase)
            all_agent_no_purchase.append(agent_no_purchase)

            temp = [pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize) for e in experts]
            temp.append(pe.get_wd(expert_no_purchase, agent_no_purchase, normalize))
            all_distances_no_purchase.append(temp)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        fig.subplots_adjust(left=0.25)
        fig.suptitle('Number of training steps: %d' % n_steps)

        columns = ['Customer {}'.format(i + 1) for i in range(n_experts)]
        columns.append('Avg. customer')
        index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

        all_distances_no_purchase = pd.DataFrame(all_distances_no_purchase, columns=columns, index=index)
        seaborn.heatmap(all_distances_no_purchase, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': "Earth mover's distance"})
        fig.suptitle('Comparison at individual level')

        if show_info: fig.text(0.5, 0.025, info, ha='center')
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'ind_' + str(n_steps) + ending_png))

        plt.close(fig)

    # Plot Wasserstein distance
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.20)
    
    n_episodes = [int(get_key_from_path(x) / args['episode_length'])  for x in model_dir_paths]
    
    ax1.plot(n_episodes, training_purchase)
    ax1.set_xlabel('Number of episodes')
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_ylabel('EMD')
    ax1.set_title('Purchase')
    
    ax2.plot(n_episodes, training_no_purcahse)
    ax2.set_xlabel('Number of episodes')
    ax2.xaxis.set_tick_params(rotation=90)
    ax2.set_ylabel('EMD')
    ax2.set_title('No purchase')

    if show_info: fig.text(0.5, 0.025, info, ha='center')
    save_plt_as_png(fig, path=join(dir_path, 'figs', 'EMD_vs_episodes.png'))
    if not show_plots: plt.close(fig)

    if show_plots: plt.show()

if __name__ == '__main__':
    eval_training()
