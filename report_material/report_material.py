import os
import sys 
import gym
import json
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance as wd

sys.path.append('../')
sys.path.append('../customer_behaviour/tools')

import policy_evaluation as pe
import results2 as res

##### ##### PARAMETERS ##### #####

dir_path = 'gail_baseline'
# dir_path = 'airl_baseline'

sample_length = 10000
n_new_customers = 50

##### ##### LOAD ARGUMENTS ##### #####

args_path = join(dir_path, 'args.txt')
args = json.loads(open(args_path, 'r').read())

##### ##### FINAL MODEL ##### #####

final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

n_experts = args['n_experts']

if args['state_rep'] == 71:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
else:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length)

# Sample customer data
customer_trajectories = env.generate_expert_trajectories(
    out_dir=None, 
    n_demos_per_expert=1,
    n_experts=n_experts+n_new_customers,
    n_expert_time_steps=sample_length
    )
customer_states = np.array(customer_trajectories['states'])
customer_actions = np.array(customer_trajectories['actions'])
customers = res.get_distribs(customer_states, customer_actions)
expert_states = np.array(customer_trajectories['states'][:n_experts])
expert_actions = np.array(customer_trajectories['actions'][:n_experts])
avg_expert = pe.get_distrib(expert_states, expert_actions)

# Sample agent data
def sample_agent_data(N, env, model, obs_normalizer):
    agent_states = []
    agent_actions = []
    for i in range(N):
        # Initialize agent with data from ith expert
        initial_state = random.choice(customer_states[i])
        states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)    
        agent_states.append(states)
        agent_actions.append(actions)
    agent_states = np.array(agent_states)
    agent_actions = np.array(agent_actions)
    return agent_states, agent_actions
# agent_states, agent_actions = sample_agent_data(n_experts, env, model, obs_normalizer)
'''
# Plot average distributions
avg_agent = pe.get_distrib(agent_states, agent_actions)

print(wd(avg_expert, avg_agent))

fig, ax = plt.subplots()
data = {'Average expert': avg_expert, 'Average agent': avg_agent}
pe.bar_plot(ax, data, colors=None, total_width=0.7)
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('AIRL')
plt.show()

# Plot some individual customers
agent2 = pe.get_distrib(agent_states[1], agent_actions[1])
expert2 = pe.get_distrib(expert_states[1], expert_actions[1])

print(wd(avg_expert, agent2))
print(wd(expert2, agent2))

fig, ax = plt.subplots()
data = {'Agent 2': agent2, 'Expert 2': expert2, 'Average expert': avg_expert}
pe.bar_plot(ax, data, colors=None, total_width=0.7)
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('AIRL')
plt.show()

agent9 = pe.get_distrib(agent_states[8], agent_actions[8])
expert9 = pe.get_distrib(expert_states[8], expert_actions[8])

print(wd(avg_expert, agent9))
print(wd(expert9, agent9))

fig, ax = plt.subplots()
data = {'Agent 9': agent9, 'Expert 9': expert9, 'Average expert': avg_expert}
pe.bar_plot(ax, data, colors=None, total_width=0.7)
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('AIRL')
plt.show()
'''
##### ##### BASELINE ##### #####

model_dir_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
model_dir_paths.sort(key=res.get_key_from_path)

data = []

for mdp in model_dir_paths:
    n_steps = res.get_key_from_path(mdp)

    print('Processing model saved after %s' % n_steps)
    
    if int(n_steps) <= 1000000: continue

    if args['state_rep'] == 71:
        env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
    else:
        env, model, obs_normalizer = pe.get_env_and_model(args, mdp, sample_length)

    agent_states, agent_actions = sample_agent_data(n_experts+n_new_customers, env, model, obs_normalizer)

    agents = res.get_distribs(agent_states, agent_actions)

    temp = []
    for i, (a, c) in enumerate(zip(agents, customers)):
        if i < n_experts:
            data.append([n_steps, wd(a, c), 'Experts'])
        else:
            data.append([n_steps, wd(a, c), 'New customers'])
        data.append([n_steps, wd(a, avg_expert), 'Average expert'])

df = pd.DataFrame(data, columns=['Number of training steps', 'Wasserstein distance', 'Comparison with'])
df.to_csv('df_gail.csv', index=False)

# df = pd.read_csv('df.csv')
# sns.set(style='darkgrid')
# g = sns.relplot(x='Number of training steps', y='Wasserstein distance', hue='Comparison with', ci=95, kind='line', data=df)
# g._legend.set_bbox_to_anchor([0.70, 0.85])
# plt.savefig('gail_vs_train.eps', format='eps', transparent=True)
# plt.show()

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
sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'})
plt.show()
'''
