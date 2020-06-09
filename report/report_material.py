import os
import sys 
import gym
import json
import random
import fnmatch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from os.path import join
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance as wd

sys.path.append('../')
sys.path.append('../customer_behaviour/tools')

# sys.path.append('customer_behaviour/tools')

import policy_evaluation as pe
import results2 as res

##### ##### PARAMETERS ##### #####

# plt.rcParams.update({'font.size': 14})

dir_path = '../report_material/gail'
# dir_path = '../report_material/gail_dummies'
# dir_path = '../report_material/gail_adams'
# dir_path = '../report_material/airl'
# dir_path = '../report_material/airl_dummies'
# dir_path = '../report_material/ail_dummies/2020-05-15_12-59-19'
# dir_path = '../report_material/case23'
# dir_path = '../report_material/ail_adams/2020-05-15_12-57-15'

sample_length = 10000
n_new_customers = 0 # 50

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

def sample_agent_data(N, args, env, model, obs_normalizer, customers, customer_states):
    agent_states = []
    agent_actions = []

    closest_expert = N * [0]

    for i in range(N):
        # Initialize agent with data from ith expert
        initial_state = random.choice(customer_states[i])
        if args['state_rep'] == 22 or args['state_rep'] == 221 or args['state_rep'] == 23 and i >= n_experts:
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

args_path = join(dir_path, 'args.txt')
args = json.loads(open(args_path, 'r').read())

n_experts = args['n_experts']

final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)
if args['state_rep'] == 71 or args['state_rep'] == 81:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
else:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length)

# Sample expert data
customer_states, customer_actions = sample_customer_data(env, n_experts, sample_length, n_new_customers)
expert_states = np.array(customer_states[:n_experts])
expert_actions = np.array(customer_actions[:n_experts])

customers = res.get_distribs(customer_states, customer_actions)
experts = customers[:n_experts]
avg_expert = pe.get_distrib(expert_states, expert_actions)

customers_purchase = []
customers_no_purchase = []
for cs, ca in zip(customer_states, customer_actions):
    temp_purchase, temp_no_purchase, _ = pe.get_cond_distribs(
        [cs], 
        [ca],
        n_last_days=7, 
        max_n_purchases=2, 
        normalize=True,
        case=args['state_rep']
        )
    customers_purchase.append(temp_purchase)
    customers_no_purchase.append(temp_no_purchase)
avg_expert_purchase, avg_expert_no_purchase, _ = pe.get_cond_distribs(
    expert_states, 
    expert_actions, 
    n_last_days=7, 
    max_n_purchases=2, 
    normalize=True,
    case=args['state_rep']
    )

# Sample agent data
agent_states, agent_actions, _ = sample_agent_data(n_experts+n_new_customers, args, env, model, obs_normalizer, customers, customer_states)

# Plot average distributions
avg_agent = pe.get_distrib(agent_states, agent_actions)
avg_agent_purchase, avg_agent_no_purchase, _ = pe.get_cond_distribs(
    agent_states, 
    agent_actions, 
    n_last_days=7, 
    max_n_purchases=2, 
    normalize=True,
    case=args['state_rep']
    )

fig, ax = plt.subplots()
# fig.suptitle('AIRL')
data = {'Average expert prediction': avg_agent_purchase, 'Average expert': avg_expert_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, legend=True, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | Purchase today')
# print('Purchase (avgerage expert): %f' % wd(avg_agent_purchase, avg_expert_purchase))

fig, ax = plt.subplots()
# fig.suptitle('AIRL')
data = {'Average expert prediction': avg_agent_no_purchase, 'Average expert': avg_expert_no_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, legend=True, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | No purchase today')
# print('No purchase (avgerage expert): %f' % wd(avg_expert_no_purchase, avg_agent_no_purchase))

print('Avgerage expert): %f' % wd(avg_agent, avg_expert))

plt.show()

# Compare against expert 2
agent2_purchase, agent2_no_purchase, _ = pe.get_cond_distribs(
    [agent_states[1]], 
    [agent_actions[1]], 
    n_last_days=7, 
    max_n_purchases=2, 
    normalize=True,
    case=args['state_rep']
    )
expert2_purchase = customers_purchase[1]
expert2_no_purchase = customers_no_purchase[1]

fig, ax = plt.subplots()
fig.suptitle('GAIL')
data = {'Prediction': agent2_purchase, 'New customer': expert2_purchase, 'Average expert': avg_expert_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | Purchase today')
print('Purchase (expert 2): %f' % wd(agent2_purchase, expert2_purchase))

fig, ax = plt.subplots()
fig.suptitle('GAIL')
data = {'Prediction': agent2_no_purchase, 'New customer': expert2_no_purchase, 'Average expert': avg_expert_no_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | No purchase today')
print('No purchase (expert 2): %f' % wd(agent2_purchase, expert2_purchase))

print('Avgerage expert): %f' % wd(pe.get_distrib(agent_states[1], agent_actions[1]), avg_expert))
print('Expert 2): %f' % wd(pe.get_distrib(agent_states[1], agent_actions[1]), customers[1]))

plt.show()

# Compare against expert 9
agent9_purchase, agent9_no_purchase, _ = pe.get_cond_distribs(
    [agent_states[8]], 
    [agent_actions[8]], 
    n_last_days=7, 
    max_n_purchases=2, 
    normalize=True,
    case=args['state_rep']
    )
expert9_purchase = customers_purchase[8]
expert9_no_purchase = customers_no_purchase[8]

fig, ax = plt.subplots()
fig.suptitle('AIRL')
data = {'Agent 9': agent9_purchase, 'Expert 9': expert9_purchase, 'Average expert': avg_expert_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | Purchase today')

fig, ax = plt.subplots()
fig.suptitle('AIRL')
data = {'Agent 9': agent9_no_purchase, 'Expert 9': expert9_no_purchase, 'Average expert': avg_expert_no_purchase}
pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
ax.set_xticks([], [])
ax.set_ylabel('Probability')
ax.set_title('Last week | No purchase today')

plt.show()

# Plot heatmap
agents = res.get_distribs(agent_states, agent_actions)
distances = []
for i, a in enumerate(agents):
    if i >= 10: continue
    temp = [wd(a, e) for e in experts]
    temp.append(wd(a, avg_expert))
    distances.append(temp)

columns = ['Expert {}'.format(i + 1) for i in range(10)]
columns.append('Average expert')
index = ['Agent {}'.format(i + 1) for i in range(10)]

df = pd.DataFrame(distances, columns=columns, index=index)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.25)
sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'}, square=True)
ax.set_title('GAIL')
plt.show()

# Plot histogram
if args['state_rep'] == 23:
    expert_amounts = np.ravel(expert_actions)[np.flatnonzero(expert_actions)]
    agent_amounts = np.ravel(agent_actions)[np.flatnonzero(agent_actions)]

    fig, ax = plt.subplots()
    ax.hist(expert_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Average expert')
    ax.hist(agent_amounts, bins=np.arange(1, 11), alpha=0.8, density=True, label='Average prediction')
    ax.set_title('AIRL')
    ax.set_xlabel('Categorized purchase amounts')
    ax.set_ylabel('Normalized frequency')
    ax.legend()
    plt.show()

##### ##### TRAINING PROCESS ##### #####
'''
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

        agent_states, agent_actions, closest_expert = sample_agent_data(n_experts+n_new_customers, args, env, model, obs_normalizer, customers, customer_states)

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
'''
##### ##### PLOT TRAINING PROCESS ##### #####
'''
baseline = 'ail_dummies'
baseline_name = 'OHE'

algo = 'gail'
algo_name = 'Basic'

n_training_steps = 20000

dfs_baseline = [] # [pd.read_csv(join(os.getcwd(), baseline, x)) for x in os.listdir(join(os.getcwd(), baseline)) if fnmatch.fnmatch(x, 'df_' + baseline + '?.csv')]
dfs_algo = [pd.read_csv(join(os.getcwd(), algo, x)) for x in os.listdir(join(os.getcwd(), algo)) if fnmatch.fnmatch(x, 'df_' + algo + '?.csv')]

for df in dfs_baseline:
    df['Algorithm'] = len(df.index) * [baseline_name]

    for x in ['Closest expert', 'Average expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

for df in dfs_algo:
    df['Algorithm'] = len(df.index) * [algo_name]

    for x in ['Closest expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

df = pd.concat(tuple(dfs_algo + dfs_baseline))
df['Number of training episodes'] = [x / 1095 for x in df['Number of training steps'].values.tolist()]
df = df[df['Number of training episodes'] <= n_training_steps]

# sel_df_baseline = df[(df['Algorithm'] == baseline_name) & (df['Number of training episodes'] == 10000) & (df['Comparison with'] == 'New customers')]
# avg_baseline = sel_df_baseline['Wasserstein distance'].mean()

# sel_df_algo = df[(df['Algorithm'] == algo_name) & (df['Number of training episodes'] == 10000) & (df['Comparison with'] == 'New customers')]
# avg_algo = sel_df_algo['Wasserstein distance'].mean()

# print(avg_algo / avg_baseline)

sns.set(style='darkgrid')
g = sns.relplot(x='Number of training episodes', y='Wasserstein distance', hue='Comparison with', ci=95, kind='line', data=df,  \
    facet_kws={'legend_out': False}) # , style='Algorithm') #, hue_order=['Experts', 'New customers', 'Average expert', 'Closest expert'])
g.fig.subplots_adjust(top=0.95)
ax = g.axes[0][0]
ax.set_ylim([0, 0.01])
ax.set_title('GAIL')
plt.legend(loc='upper right')
plt.show()
'''
##### ##### GAIL / AIRL / AIL-MSD ##### #####
'''
basic = 'ail'
basic_name = 'Basic'

dummies = 'ail_dummies'
dummies_name = 'OHE'

adams = 'ail_adams'
adams_name = 'CS days'

n_training_steps = 20000

dfs_basic = [pd.read_csv(join(os.getcwd(), basic, x)) for x in os.listdir(join(os.getcwd(), basic)) if fnmatch.fnmatch(x, 'df_' + basic + '?.csv')]
dfs_dummies = [pd.read_csv(join(os.getcwd(), dummies, x)) for x in os.listdir(join(os.getcwd(), dummies)) if fnmatch.fnmatch(x, 'df_' + dummies + '?.csv')]
dfs_adams = [pd.read_csv(join(os.getcwd(), adams, x)) for x in os.listdir(join(os.getcwd(), adams)) if fnmatch.fnmatch(x, 'df_' + adams + '?.csv')]

for df in dfs_basic:
    df['State representation'] = len(df.index) * [basic_name]
    for x in ['Closest expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

for df in dfs_dummies:
    df['State representation'] = len(df.index) * [dummies_name]
    for x in ['Closest expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

for df in dfs_adams:
    df['State representation'] = len(df.index) * [adams_name]
    for x in ['Closest expert', 'Experts']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

df = pd.concat(tuple(dfs_basic + dfs_dummies + dfs_adams))
df['Number of training episodes'] = [x / 1095 for x in df['Number of training steps'].values.tolist()]
df = df[df['Number of training episodes'] <= n_training_steps]

sns.set(style='darkgrid')
g = sns.relplot(x='Number of training episodes', y='Wasserstein distance', hue='State representation', ci=95, kind='line', data=df,  \
    facet_kws={'legend_out': False})
g.fig.subplots_adjust(top=0.95)
ax = g.axes[0][0]
ax.set_ylim([0, 0.01])
ax.set_title('MMCT-GAIL')
# plt.legend(loc='upper right')
plt.show()
'''
##### ##### PURCHASE AMOUNTS ##### #####
'''
n_training_steps = 20000

dfs = [pd.read_csv(join(os.getcwd(), 'case23', x)) for x in os.listdir(join(os.getcwd(), 'case23')) if fnmatch.fnmatch(x, 'df_23_' + '?.csv')]

for df in dfs:
    df['Algorithm'] = len(df.index) * ['AIL-MSD']

    for x in ['Closest expert']:
        indices = df[df['Comparison with'] == x].index
        df.drop(indices, inplace=True)

df = pd.concat(tuple(dfs))
df['Number of training episodes'] = [x / 1095 for x in df['Number of training steps'].values.tolist()]
df = df[df['Number of training episodes'] <= n_training_steps]

sns.set(style='darkgrid')
g = sns.relplot(x='Number of training episodes', y='Wasserstein distance', hue='Comparison with', ci=95, kind='line', data=df,  \
    facet_kws={'legend_out': False})
g.fig.subplots_adjust(top=0.95)
ax = g.axes[0][0]
ax.set_ylim([0, 0.04])
ax.set_title('AIL-MSD')
plt.legend(loc='lower right')
plt.show()
'''
##### ##### HEATMAPS ##### #####
'''
args_path = join(dir_path, 'args.txt')
args = json.loads(open(args_path, 'r').read())

n_experts = args['n_experts']

model_dir_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
model_dir_paths.sort(key=res.get_key_from_path)

columns = ['Expert {}'.format(i + 1) for i in range(n_experts)]
index = ['Agent {}'.format(i + 1) for i in range(n_experts)]

for mdp in model_dir_paths:
    n_steps = res.get_key_from_path(mdp)

    print('Processing model saved after %s' % n_steps)

    if int(n_steps) <= 4000000: continue

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
    fig.suptitle('AIL-MSD')
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
args_path = join(dir_path, 'args.txt')
args = json.loads(open(args_path, 'r').read())

n_experts = 10

final_model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)
if args['state_rep'] == 71 or args['state_rep'] == 81:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length, n_experts_in_adam_basket=n_experts+n_new_customers)
else:
    env, model, obs_normalizer = pe.get_env_and_model(args, final_model_dir_path, sample_length)

# Sample expert data
expert_states, expert_actions = sample_customer_data(env, n_experts, sample_length, n_new_customers)

experts_purchase = []
experts_no_purchase = []
for cs, ca in zip(customer_states, customer_actions):
    temp_purchase, temp_no_purchase, _ = pe.get_cond_distribs(
        [cs], 
        [ca],
        n_last_days=7, 
        max_n_purchases=2, 
        normalize=True,
        case=args['state_rep']
        )
    experts_purchase.append(temp_purchase)
    experts_no_purchase.append(temp_no_purchase)

for i, (e_p, e_np) in enumerate(zip(experts_purchase, experts_no_purchase)):
    fig, ax = plt.subplots()
    data = {'Expert {}'.format(i+1): e_p}
    pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
    ax.set_xticks([], [])
    ax.set_ylabel('Probability')
    ax.set_title('Last week | Purchase today')

    fig, ax = plt.subplots()
    data = {'Expert {}'.format(i+1): e_np}
    pe.bar_plot(ax, data, colors=None, total_width=0.9, loc='upper center')
    ax.set_xticks([], [])
    ax.set_ylabel('Probability')
    ax.set_title('Last week | No purchase today')

plt.show()

# labels = ['Expert {}'.format(i + 1) for i in range(n_experts)]

# df = pd.DataFrame(squareform(pdist(np.array(experts), lambda u, v: wd(u, v))), columns=labels, index=labels)

# fig, ax = plt.subplots()
# fig.subplots_adjust(bottom=0.16)
# sns.heatmap(df, cmap='BuPu', ax=ax, linewidth=1, cbar_kws={'label': 'Wasserstein distance'}, square=True)
# plt.show()
'''
