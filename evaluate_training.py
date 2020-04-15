import os
import json
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import customer_behaviour.tools.policy_evaluation as pe
from os.path import join
from customer_behaviour.tools.tools import save_plt_as_png
from evaluate_policy import Expert
from scipy.cluster.hierarchy import fclusterdata
from scipy.stats import wasserstein_distance
from matplotlib.ticker import MaxNLocator

# dir_path = 'results_anton/2020-03-10_14-04-29'  # 10 experts | 96 historical events  | length_expert_TS = 256  | 5 000 episodes
# dir_path = 'results_anton/2020-03-12_11-14-55'  # 10 experts | 256 historical events | length_expert_TS = 256  | 10 000 episodes
# dir_path = 'results_anton/2020-03-16_20-13-19'  # 10 experts | 256 historical events | length_expert_TS = 1024 | 15 000 episodes
# dir_path = 'results_anton/2020-03-18_14-01-33'  # 10 experts | 96 historical events  | length_expert_TS = 256  | 5 000 episodes
# dir_path = 'results_anton/2020-03-17_11-59-59'  # 10 experts | state_rep = 11 (sex and age)
# dir_path = 'results_anton/2020-03-19_09-56-34'  # 10 experts | 96 historical events  | length_expert_TS = 256  | 10 000 episodes
# dir_path = 'results_anton/2020-03-20_12-52-09'  # 10 experts | state_rep = 22 | 100 historical events | length_expert_TS = 256  | 15 000 episodes | norm_obs = False
# dir_path = 'results_anton/2020-03-20_12-47-08'  # 10 experts | state_rep = 22 | 50 historical events | length_expert_TS = 256  | 15 000 episodes  | norm_obs = True
# dir_path = 'results_anton/2020-03-20_12-51-37'  # 10 experts | state_rep = 22 | 100 historical events | length_expert_TS = 256  | 15 000 episodes  | norm_obs = True
# dir_path = 'results_anton/2020-03-27_09-37-13'  # 10 experts | state_rep = 22 | 100 historical events | length_expert_TS = 256  | 20 000 episodes | norm_obs = False
# dir_path = 'results_anton/2020-03-27_09-38-46'  # 10 experts | state_rep = 22 | 100 historical events | length_expert_TS = 512  | 20 000 episodes | norm_obs = False
# dir_path = 'results_anton/2020-03-27_09-39-34'  # 10 experts | state_rep = 22 | 100 historical events | length_expert_TS = 1024 | 20 000 episodes | norm_obs = False

# Mode collapse
# dir_path = 'mode_collapse/2020-04-07_09-24-54'  # state_rep = 24 | episodes = 15000 | length_expert_TS = 256 | show_dummies_D = False
# dir_path = 'mode_collapse/2020-04-07_09-24-37'  # state_rep = 24 | episodes = 15000 | length_expert_TS = 256 | show_dummies_D = False
# dir_path = 'mode_collapse/2020-04-07_09-25-12'  # state_rep = 24 | episodes = 15000 | length_expert_TS = 256 | show_dummies_D = False
# dir_path = 'mode_collapse/2020-04-08_09-39-53'  # state_rep = 24 | episodes = 15000 | length_expert_TS = 256 | show_dummies_D = True 
# dir_path = 'mode_collapse/2020-04-08_09-40-34'  # state_rep = 24 | episodes = 15000 | length_expert_TS = 256 | show_dummies_D = True
# dir_path = 'mode_collapse/2020-04-09_11-57-50'  # state_rep = 22 | n_experts = 3
# dir_path = 'mode_collapse/2020-04-09_11-58-13'  # state_rep = 22 | n_experts = 4
# dir_path = 'mode_collapse/2020-04-09_11-58-41'  # state_rep = 22 | n_experts = 5

# Case 23
# dir_path = 'case23/2020-04-07_21-13-59'  # PAC_k = 2 | gamma = 0.0 | 30000 episodes
# dir_path = 'case23/2020-04-07_21-11-34'  # PAC_k = 1 | gamma = 0.0 | 20000 episodes
# dir_path = 'case23/2020-04-07_21-13-26'  # PAC_k = 2 | gamma = 0.0 | 20000 episodes
# dir_path = 'case23/2020-04-07_21-12-43'  # PAC_k = 1 | gamma = 0.0 | 30000 episodes
# dir_path = 'case23/2020-04-09_11-11-53'  # PAC_k = 1 | gamma = 0.0 | 40000 episodes | D = 4 * [64] | G = 4 * [64]

# Case 31
# dir_path = 'case31/2020-04-08_14-47-28'  # show_dummies_D = False
# dir_path = 'case31/2020-04-09_11-59-25'  # show_dummies_D = True
# dir_path = 'case31/2020-04-11_10-55-36'  # show_dummies_D = True

# Case 4
# dir_path = 'case4/2020-04-09_12-00-51'



normalize = True
n_last_days = 7
max_n_purchases_per_n_last_days = 2
show_info = True
save_plots = True
cluster_dist = 0.006

def main():
    # Load arguments
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    if show_info: info = pe.get_info(args)
    n_sampled_days = args['eval_episode_length'] * args['n_experts']

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)
    ending_png = '_normalize.png' if normalize else '.png'

    # Helper function
    def get_key_from_path(path):
        temp = path.split('_')
        temp = int(temp[-1].split('.')[0])  # Adam steps
        steps = int((temp / args['epochs']) * args['batchsize'])
        return steps

    # Load expert data
    expert_data_path = join(dir_path, 'eval_expert_trajectories.npz')
    expert_states, expert_actions = load_data(expert_data_path)

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

    # Load agent data
    agent_data_dir_path = join(dir_path, 'states_actions')
    agent_data_paths = os.listdir(agent_data_dir_path)
    agent_data_paths = [join(agent_data_dir_path, x) for x in agent_data_paths if x.endswith('npz')]
    agent_data_paths.sort(key=get_key_from_path)

    training_purchase = []
    training_no_purcahse = []

    training_n_clusters_purchase = []
    training_n_clusters_no_purchase = []

    for adp in agent_data_paths:
        n_updates = get_key_from_path(adp)
        agent_states, agent_actions = load_data(adp)

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
        
        agent_shopping_ratio = format(agent_n_shopping_days / n_sampled_days, '.3f')
        expert_shopping_ratio = format(expert_n_shopping_days / n_sampled_days, '.3f')
        expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
        agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Number of traning steps: %d' % n_updates)

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
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'pop_' + str(n_updates) + ending_png))

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
        fig.suptitle('Number of traning steps: %d' % n_updates)

        columns = ['Customer {}'.format(i + 1) for i in range(n_experts)]
        columns.append('Avg. customer')
        index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

        all_distances_no_purchase = pd.DataFrame(all_distances_no_purchase, columns=columns, index=index)
        seaborn.heatmap(all_distances_no_purchase, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': "Earth mover's distance"})
        fig.suptitle('Comparison at individual level')

        if show_info: fig.text(0.5, 0.025, info, ha='center')
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', 'ind_' + str(n_updates) + ending_png))

        plt.close(fig)

        # Cluster expert data (purcase)
        X = np.array(all_agent_purchase)
        T_purchase = fclusterdata(X, cluster_dist, 'distance', method='single', metric=lambda u, v: wasserstein_distance(u, v))
        training_n_clusters_purchase.append(len(set(T_purchase)))

        # Cluster expert data (no purcase)
        X = np.array(all_agent_no_purchase)
        T_no_purchase = fclusterdata(X, cluster_dist, 'distance', method='single', metric=lambda u, v: wasserstein_distance(u, v))
        training_n_clusters_no_purchase.append(len(set(T_no_purchase)))
        print(T_no_purchase)

    # Plot Wasserstein distance
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.20)
    
    n_episodes = [int(get_key_from_path(x) / args['episode_length'])  for x in agent_data_paths]
    
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

    show_info: fig.text(0.5, 0.025, info, ha='center')
    save_plt_as_png(fig, path=join(dir_path, 'figs', 'EMD_vs_episodes.png'))

    plt.show()

    # Plot number of clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.20)
    fig.suptitle('Max cophenetic distance: %f' % cluster_dist)
    
    ax1.plot(n_episodes, training_n_clusters_purchase)
    ax1.set_xlabel('Number of episodes')
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_ylabel('Number of clusters')
    ax1.set_title('Purchase')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax2.plot(n_episodes, training_n_clusters_no_purchase)
    ax2.set_xlabel('Number of episodes')
    ax2.xaxis.set_tick_params(rotation=90)
    ax2.set_ylabel('Number of clusters')
    ax2.set_title('No purchase')
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    show_info: fig.text(0.5, 0.025, info, ha='center')
    save_plt_as_png(fig, path=join(dir_path, 'figs', 'n_clusters_vs_episodes.png'))

    plt.show()

def load_data(file):
    data = np.load(file, allow_pickle=True)
    assert sorted(data.files) == sorted(['states', 'actions'])
    states = data['states']
    actions = data['actions']
    return states, actions

if __name__ == '__main__':
    main()