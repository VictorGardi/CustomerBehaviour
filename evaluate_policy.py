import os
import gym
import custom_gym
import chainer
import chainerrl
import json
import itertools
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import customer_behaviour.tools.policy_evaluation as pe
from os.path import join
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from customer_behaviour.tools.tools import save_plt_as_eps, save_plt_as_png

# dir_path = 'saved_results/gail/discrete_events/1_expert(s)/case_21/monday_0309/2020-03-09_10-26-19'
# dir_path = 'saved_results/gail/discrete_events/10_expert(s)/case_21/tuesday_0310/2020-03-10_14-04-29'
dir_path = 'results_anton/2020-03-12_11-14-55'  # 10 experts | 256 historical events | 10 000 episodes

sample_length = 10000
normalize = True
n_demos_per_expert = 10
n_last_days = 7
max_n_purchases_per_n_last_days = 2

def main():
    # Load arguments
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    # evaluate_policy_at_population_level(args, model_dir_path)
    # evaluate_policy_at_individual_level(args, model_dir_path)
    compare_clusters(args, model_dir_path)

############################
############################

class Expert():
    def __init__(self, purchases, no_purchases, avg_purchase, avg_no_purchase):
        self.purchases = purchases
        self.no_purchases = no_purchases

        self.avg_purchase = avg_purchase
        self.avg_no_purchase = avg_no_purchase

        self.calculate_pairwise_distances()

    def calculate_pairwise_distances(self):
        self.distances_purchase = []
        for u, v in itertools.combinations(self.purchases, 2):
            wd = pe.get_wd(u, v, normalize)
            self.distances_purchase.append(wd)

        self.distances_no_purchase = []
        for u, v in itertools.combinations(self.no_purchases, 2):
            wd = pe.get_wd(u, v, normalize)
            self.distances_no_purchase.append(wd)

def compare_clusters(args, model_dir_path):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Get multiple samples from each expert
    assert (sample_length % n_demos_per_expert) == 0
    N = int(sample_length / n_demos_per_expert)
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=n_demos_per_expert, n_expert_time_steps=N)
    expert_states = np.array(expert_trajectories['states'])
    expert_actions = np.array(expert_trajectories['actions'])
    sex = expert_trajectories['sex']
    age = expert_trajectories['age']

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

    for i in range(n_experts):
        # Sample agent data starting with expert's history
        initial_state = expert_states[i][0]
        agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
            [agent_states], 
            [agent_actions], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        # Compare distributions (purchase)
        temp = [1000 * pe.get_wd(e.avg_purchase, agent_purchase, normalize) for e in experts]  # [mW]
        temp.append(1000 * pe.get_wd(avg_expert_purchase, agent_purchase, normalize))
        distances_purchase.append(temp)

        # Compare distributions (no purchase)
        temp = [1000 * pe.get_wd(e.avg_no_purchase, agent_no_purchase, normalize) for e in experts]  # [mW]
        temp.append(1000 * pe.get_wd(avg_expert_no_purchase, agent_no_purchase, normalize))
        distances_no_purchase.append(temp)

    columns = ['Expert {}'.format(i + 1) for i in range(n_experts)]
    columns.append('Average expert')
    index = ['Expert {}'.format(i + 1) for i in range(n_experts)]

    distances_purchase = pd.DataFrame(distances_purchase,
        columns=columns,
        index=index)

    distances_no_purchase = pd.DataFrame(distances_no_purchase,
        columns=columns,
        index=index)

    fig, ax = plt.subplots()
    seaborn.heatmap(
        distances_purchase,
        cmap='OrRd',
        ax=ax,
        linewidth=1,
        cbar_kws={'label': 'mW'}
        )
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle('Purchase')

    fig, ax = plt.subplots()
    seaborn.heatmap(
        distances_no_purchase,
        cmap='OrRd',
        ax=ax,
        linewidth=1,
        cbar_kws={'label': 'mW'}
        )
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle('No purchase')

    plt.show()


'''
# Heatmap
avg_purchase = np.array([expert.avg_purchase for expert in experts])

df = pd.DataFrame(squareform(pdist(avg_purchase, lambda u, v: wasserstein_distance(u, v))),
    columns=['Expert {}'.format(i + 1) for i in range(n_experts)],
    index=['Expert {} | {} | {}'.format(i + 1, sex[i], age[i]) for i in range(n_experts)]
    )

_, ax = plt.subplots()
seaborn.heatmap(
    df,
    cmap='OrRd',
    ax=ax,
    linewidth=1
)
plt.show()
'''

############################
############################

def evaluate_policy_at_individual_level(args, model_dir_path):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample expert data to calculate average expert behavior
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

    n_experts = args['n_experts']

    for i in range(n_experts):
        # Sample agent data starting with expert's history
        initial_state = expert_states[i][0]
        agent_states, agent_actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)

        agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
            [agent_states], 
            [agent_actions], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
            [expert_states[i]], 
            [expert_actions[i]], 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        # Calculate Wasserstein distances
        wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
        wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)

        os.makedirs(join(dir_path, 'figs'), exist_ok=True)

        # Plot (purchase)
        fig, ax = plt.subplots()
        expert_str = 'Expert (' + str(expert_n_shopping_days) + ')'
        agent_str = 'Agent (' + str(agent_n_shopping_days) + ')'
        data = {expert_str: expert_purchase, agent_str: agent_purchase, 'Average expert': avg_expert_purchase}
        pe.bar_plot(ax, data, colors=None, total_width=0.7)
        pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
        fig.subplots_adjust(bottom=0.3)
        fig.suptitle('Purchase')
        ax.set_title('Wasserstein distance: {0:.10f}'.format(wd_purchase))

        # Plot (no purchase)
        fig, ax = plt.subplots()
        expert_str = 'Expert (' + str(sample_length - expert_n_shopping_days) + ')'
        agent_str = 'Agent (' + str(sample_length - agent_n_shopping_days) + ')'
        data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase, 'Average expert': avg_expert_no_purchase}
        pe.bar_plot(ax, data, colors=None, total_width=0.7)
        pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
        fig.subplots_adjust(bottom=0.3)
        fig.suptitle('No purchase')
        ax.set_title('Wasserstein distance: {0:.10f}'.format(wd_no_purchase))

        plt.show()

############################
############################

def evaluate_policy_at_population_level(args, model_dir_path):
    # Load environment, model and observation normalizer
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length)

    # Get possible validation states
    possible_val_states = pe.get_possible_val_states(n_last_days, max_n_purchases_per_n_last_days)

    # Sample agent data
    agent_states = []
    agent_actions = []
    for i in range(args['n_experts']):
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

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)
    
    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    n_sampled_days = sample_length * args['n_experts']

    # Plot (purchase)
    fig, ax = plt.subplots()
    expert_str = 'Expert (' + str(expert_n_shopping_days) + ')'
    agent_str = 'Agent (' + str(agent_n_shopping_days) + ')'
    data = {expert_str: expert_purchase, agent_str: agent_purchase}
    pe.bar_plot(ax, data, colors=None, total_width=0.7)
    pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle('Purchase')
    ax.set_title('Wasserstein distance: {0:.10f}'.format(wd_purchase))
    # save_plt_as_eps(fig, path=join(dir_path, 'figs', 'purchase' + ending_eps))
    # save_plt_as_png(fig, path=join(dit_path, 'figs', 'purchase' + ending_png))

    # Plot (no purchase)
    fig, ax = plt.subplots()
    expert_str = 'Expert (' + str(n_sampled_days - expert_n_shopping_days) + ')'
    agent_str = 'Agent (' + str(n_sampled_days - agent_n_shopping_days) + ')'
    data = {expert_str: expert_no_purchase, agent_str: agent_no_purchase}
    pe.bar_plot(ax, data, colors=None, total_width=0.7)
    pe.set_xticks(ax, possible_val_states, max_n_purchases_per_n_last_days)
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle('No purchase')
    ax.set_title('Wasserstein distance: {0:.10f}'.format(wd_no_purchase))
    # save_plt_as_eps(fig, path=join(dir_path, 'figs', 'no_purchase' + ending_eps))
    # save_plt_as_png(fig, path=join(dit_path, 'figs', 'no_purchase' + ending_png))

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
