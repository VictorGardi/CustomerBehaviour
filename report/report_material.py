import os
import sys 
import gym
import json
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from os.path import join
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance as wd

# sys.path.append('../')
# sys.path.append('../customer_behaviour/tools')

sys.path.append('customer_behaviour/tools')

import policy_evaluation as pe
import results2 as res

##### ##### PARAMETERS ##### #####

# dir_path = 'gail'
# dir_path = 'airl'
# dir_path = 'ail-md_dummy'
# dir_path = 'gail_dummy' 
# dir_path = '71_100'
# dir_path = '71_100novel'

sample_length = 10000
n_new_customers = 50

##### ##### HELPER FUNCTIONS ##### #####

def sample_customer_data(env, n_experts, sample_length=10000, n_new_customers=50):
    customer_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=1,
        n_experts=n_experts+n_new_customers,
        n_expert_time_steps=sample_length
        )
    customer_states = np.array(customer_trajectories['states'])
    customer_actions = np.array(customer_trajectories['actions'])

    return customer_states, customer_actions

def sample_agent_data(N, args, env, model, obs_normalizer, customers, customer_states, experts):
    agent_states = []
    agent_actions = []

    closest_expert = N * [0]

    for i in range(N):
        # Initialize agent with data from ith expert
        initial_state = random.choice(customer_states[i])
        if args['state_rep'] == 22 or args['state_rep'] == 221 or args['state_rep'] == 23 and i >= args['n_experts']:
            # Find closest expert
            c = customers[i]
            distances = [wd(c, e) for e in experts]
            dummy = np.argsort(distances)[0]
            closest_expert[i] = dummy
            initial_state[dummy] = 1
        states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)    
        agent_states.append(states)
        agent_actions.append(actions)
    agent_states = np.array(agent_states)
    agent_actions = np.array(agent_actions)
    return agent_states, agent_actions, closest_expert

##### ##### FINAL MODEL ##### #####
'''
args_path = join(dir_path, 'args.txt')
args = json.loads(open(args_path, 'r').read())

n_experts = args['n_experts']

final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)
if args['state_rep'] == 71 or args['state_rep'] == 81:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
else:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length)

customer_states, customer_actions = sample_customer_data(env, n_experts, sample_length, n_new_customers)
customers = res.get_distribs(customer_states, customer_actions)
expert_states = np.array(customer_trajectories['states'][:n_experts])
expert_actions = np.array(customer_trajectories['actions'][:n_experts])
experts = customers[:n_experts]
avg_expert = pe.get_distrib(expert_states, expert_actions)

# Sample agent data
agent_states, agent_actions, _ = sample_agent_data(n_experts+n_new_customers, args, env, model, obs_normalizer, customers, customer_states)

# Plot average distributions
avg_agent = pe.get_distrib(agent_states, agent_actions)

# plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots()
data = {'Average expert': avg_expert, 'Average agent': avg_agent}
pe.bar_plot(ax, data, colors=None, total_width=0.9, legend=True, loc='upper center')
ax.set_xticks([], [])
# loc = plticker.MultipleLocator(base=0.03)
# ax.yaxis.set_major_locator(loc)
ax.set_ylabel('Probability')
ax.set_title('GAIL')

plt.show()

# Plot some individual customers
agent2 = pe.get_distrib(agent_states[1], agent_actions[1])
expert2 = pe.get_distrib(expert_states[1], expert_actions[1])

fig, ax = plt.subplots()
data = {'Agent 2': agent2, 'Expert 2': expert2, 'Average expert': avg_expert}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('GAIL')
plt.show()

agent9 = pe.get_distrib(agent_states[8], agent_actions[8])
expert9 = pe.get_distrib(expert_states[8], expert_actions[8])

# print(wd(avg_expert, agent9))
# print(wd(expert9, agent9))

fig, ax = plt.subplots()
data = {'Agent 9': agent9, 'Expert 9': expert9, 'Average expert': avg_expert}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('GAIL')
plt.show()

# Plot heatmap
agents = res.get_distribs(agent_states, agent_actions)
distances = []
# distances2 = []
for i, a in enumerate(agents):
    if i >= 10: continue
    temp = [wd(a, e) for e in experts]
    # temp.append(wd(a, avg_expert))
    distances.append(temp)
    # distances2.append(wd(a, experts[i]))

# # print(distances2)
# # print(np.mean(distances2))

columns = ['Expert {}'.format(i + 1) for i in range(10)]
# columns.append('Avg. expert')
index = ['Agent {}'.format(i + 1) for i in range(10)]

df = pd.DataFrame(distances, columns=columns, index=index)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'}, square=True)
ax.set_title('GAIL')
plt.show()
'''
##### ##### TRAINING PROCESS ##### #####

def save_df(dir_path, folder_name, sample_length=10000, n_new_customers=50):
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    n_experts = args['n_experts']

    final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)
    if args['state_rep'] == 71 or args['state_rep'] == 81:
        env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
    else:
        env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length)

    customer_states, customer_actions = sample_customer_data(env, n_experts, sample_length, n_new_customers)
    customers = res.get_distribs(customer_states, customer_actions)
    expert_states = customer_states[:n_experts]
    expert_actions = customer_actions[:n_experts]
    experts = customers[:n_experts]
    avg_expert = pe.get_distrib(expert_states, expert_actions)

    model_dir_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
    model_dir_paths.sort(key=res.get_key_from_path)

    data = []
    for mdp in model_dir_paths:
        n_steps = res.get_key_from_path(mdp)

        print('Processing model saved after %s' % n_steps)
        
        if int(n_steps) <= 1000000: continue

        if args['state_rep'] == 71 or args['state_rep'] == 81:
            env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
        else:
            env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length)

        agent_states, agent_actions, closest_expert = sample_agent_data(n_experts+n_new_customers, args, env, model, obs_normalizer, customers, customer_states, experts)

        agents = res.get_distribs(agent_states, agent_actions)
        avg_agent = pe.get_distrib(agent_states[:n_experts], agent_actions[:n_experts])

        temp = []
        for i, (a, c) in enumerate(zip(agents, customers)):
            if i < n_experts:
                data.append([n_steps, wd(a, c), 'Experts'])
            else:
                data.append([n_steps, wd(a, c), 'New customers'])
                data.append([n_steps, wd(a, experts[closest_expert[i]]), 'Closest expert'])

        data.append([n_steps, wd(avg_agent, avg_expert), 'Average expert'])

    df = pd.DataFrame(data, columns=['Number of training steps', 'Wasserstein distance', 'Comparison with'])
    
    os.makedirs(join('report', folder_name), exist_ok=True) 
    counter = len([x for x in os.listdir(join('report', folder_name)) if x.endswith('.csv')])
    df.to_csv(join('report', folder_name, 'df_' + folder_name + str(counter + 1) + '.csv'), index=False)

##### ##### PLOT TRAINING PROCESS ##### #####
'''
import fnmatch

baseline = 'gail_dummies'
baseline_name = 'GAIL + dummies'

algo = 'ail-md_dummies'
algo_name = 'AIL_MSD + dummies'

n_training_steps = 10000

dfs_baseline = [pd.read_csv(x) for x in os.listdir() if x == 'df_' + baseline + '.csv' or fnmatch.fnmatch(x, 'df_' + baseline + '?.csv')]
dfs_algo = [pd.read_csv(x) for x in os.listdir() if x == 'df_' + algo + '.csv' or fnmatch.fnmatch(x, 'df_' + algo + '?.csv')]

for df in dfs_baseline:
    df['Algorithm'] = len(df.index) * [baseline_name]

    for x in ['Closest expert', 'Average expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

for df in dfs_algo:
    df['Algorithm'] = len(df.index) * [algo_name]

    for x in ['Closest expert']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

df = pd.concat(tuple(dfs_algo + dfs_baseline))
df['Number of training episodes'] = [x / 1095 for x in df['Number of training steps'].values.tolist()]
df = df[df['Number of training episodes'] <= n_training_steps]

sel_df_baseline = df[(df['Algorithm'] == baseline_name) & (df['Number of training episodes'] == 10000) & (df['Comparison with'] == 'New customers')]
avg_baseline = sel_df_baseline['Wasserstein distance'].mean()

sel_df_algo = df[(df['Algorithm'] == algo_name) & (df['Number of training episodes'] == 10000) & (df['Comparison with'] == 'New customers')]
avg_algo = sel_df_algo['Wasserstein distance'].mean()

print(avg_algo / avg_baseline)

sns.set(style='darkgrid')
g = sns.relplot(x='Number of training episodes', y='Wasserstein distance', hue='Comparison with', ci=95, kind='line', data=df,  \
    facet_kws={'legend_out': False}, style='Algorithm')
g.fig.subplots_adjust(top=0.95)
ax = g.axes[0][0]
ax.set_ylim([0, 0.01])
# ax.set_title('GAIL')
# plt.legend(loc='upper right')
plt.show()
'''
##### ##### HEATMAPS ##### #####
'''
model_dir_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
model_dir_paths.sort(key=res.get_key_from_path)

columns = ['Expert {}'.format(i + 1) for i in range(n_experts)]
index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

for mdp in model_dir_paths:
    n_steps = res.get_key_from_path(mdp)

    print('Processing model saved after %s' % n_steps)

    if int(n_steps) <= 1000000: continue

    if args['state_rep'] == 71:
        env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length, n_experts_in_adam_basket=n_experts)
    else:
        env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length)

    agent_states, agent_actions, _ = sample_agent_data(n_experts, args, env, model, obs_normalizer, customers, customer_states)

    agents = res.get_distribs(agent_states, agent_actions)
    distances = []
    for i, a in enumerate(agents):
        temp = [wd(a, e) for e in experts]
        distances.append(temp)

    df = pd.DataFrame(distances, columns=columns, index=index)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    # sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'}, square=True)
    # ax.set_title('%s training steps' % n_steps)
    sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar=False, square=True)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_title('%d training episodes' % (int(n_steps) / 1095))
    plt.show()
'''
##### ##### EXPERT VISUALIZATION ##### #####
'''
experts = [pe.get_distrib(s, a) for s, a in zip(expert_states, expert_actions)]

for i, e in enumerate(experts):
    fig, ax = plt.subplots()
    data = {'Expert {}'.format(i+1): e}
    pe.bar_plot(ax, data, colors=None, total_width=0.7)
    ax.set_xticks([], [])
    ax.set_ylabel('Probability')

labels = ['Expert {}'.format(i + 1) for i in range(n_experts)]

df = pd.DataFrame(squareform(pdist(np.array(experts), lambda u, v: wd(u, v))), columns=labels, index=labels)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.16)
sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'}, square=True)
plt.show()
'''