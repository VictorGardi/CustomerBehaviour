import os
import gym
import json
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from collections import OrderedDict
from scipy.stats import wasserstein_distance as wd
from scipy.spatial.distance import euclidean as ed

import sys 
tools_path = join(os.getcwd(), 'customer_behaviour/tools')
sys.path.insert(1, tools_path)
import policy_evaluation as pe

dir_path = 'adam_days'
param = dir_path
n_new_customers = 50
sample_length = 10000
N = 1  # The number of reinitializations when predicting a new customer 

def main():
    data = load_data()
    plot(data)

def plot(data):
    # Sample customer data
    args = json.loads(open(join([join(dir_path, x) for x in os.listdir(dir_path) if x.startswith('2020')][0], 'args.txt'), 'r').read())
    n_experts = args['n_experts']

    if args['state_rep'] == 71:
        env = pe.get_env_and_model(args, '', sample_length, only_env=True, n_experts_in_adam_basket=n_experts+n_new_customers)
    else:
        env = pe.get_env_and_model(args, '', sample_length, only_env=True)

    customer_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=1,
        n_experts=n_experts+n_new_customers,
        n_expert_time_steps=sample_length
        )
    customer_states = np.array(customer_trajectories['states'])
    customer_actions = np.array(customer_trajectories['actions'])
    customers = get_distribs(customer_states, customer_actions)

    expert_states = np.array(customer_trajectories['states'][:n_experts])
    expert_actions = np.array(customer_trajectories['actions'][:n_experts])
    avg_expert = pe.get_distrib(expert_states, expert_actions)

    param_values = list(data.keys())
    n_params = len(param_values)
    step_values = list(data[param_values[0]][0].models.keys())
    n_step_values = len(step_values)

    import seaborn as sns
    import pandas as pd

    df1 = pd.DataFrame(columns=['param_value', 'n_train_steps', 'diff_train'])
    df2 = pd.DataFrame(columns=['param_value', 'n_train_steps', 'diff_test'])

    diff_train_vs_step = [[[] for j in range(n_step_values)] for i in range(n_params)]
    diff_test_vs_step = [[[] for j in range(n_step_values)] for i in range(n_params)]

    for param_value, results in data.items():
        print('Processing parameter value {}'.format(param_value))
        for result in results:
            for n_train_steps, agent in result.models.items():
                if int(n_train_steps) <= 1000000: continue
                for i, (a, c) in enumerate(zip(agent, customers)):
                    assert len(a) == 1
                    diff = wd(a[0], c)
                    if i < n_experts:
                        df1.loc[len(df1.index)] = [get_label_from_param_value(param_value), n_train_steps, diff]
                    else:
                        df2.loc[len(df2.index)] = [get_label_from_param_value(param_value), n_train_steps, diff]
                    
    sns.relplot(x='n_train_steps', y='diff_train', hue='param_value', ci=95, kind='line', data=df1)
    sns.relplot(x='n_train_steps', y='diff_test', hue='param_value', ci=95, kind='line', data=df2)

    plt.show()

def load_data():
    data = {}
    
    # Load data
    data_paths = [join(dir_path, x) for x in os.listdir(dir_path) if x.startswith('2020')]
    data_paths.sort()

    for i, path in enumerate(data_paths):
        print('Processing folder {} of {}'.format(i + 1, len(data_paths)))

        args = json.loads(open(join(path, 'args.txt'), 'r').read())
        n_experts = args['n_experts']

        content = os.listdir(path)
        assert 'result2.pkl' in content
        result = load_result(path)

        if args[param] in data:
            data[args[param]].append(result)
        else:
            data[args[param]] = [result]

    return data

def save_data(path, sample_length, n_new_customers, N):
    args = json.loads(open(join(path, 'args.txt'), 'r').read())
  
    models = {}

    model_paths = [d for d in [x[0] for x in os.walk(path)] if d.endswith('checkpoint')]
    model_paths.sort(key=get_key_from_path)

    for mp in model_paths:
        n_train_steps = get_key_from_path(mp)
        if int(n_train_steps) <= 1000000: continue

        print('Collecting data from model saved after %s steps' % n_train_steps)

        agent = evaluate(args, mp, n_new_customers, sample_length, N)
        
        models[n_train_steps] = agent

    result = Result(models)
    save_result(result, path)

def evaluate(args, model_path, n_new_customers, sample_length, N):

    n_experts = args['n_experts']

    if args['state_rep'] == 71:
        env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False, n_experts_in_adam_basket=n_experts+n_new_customers)
    else:
        env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False)

    agents = []

    for seed in range(n_experts + n_new_customers):
        temp_agents = []

        if args['state_rep'] == 71: 
            adam_basket = np.random.permutation(env.case.adam_baskets[seed])
            env.case.i_expert = seed

        env.model.spawn_new_customer(seed)
        sample = env.case.get_sample(
            n_demos_per_expert=1,
            n_historical_events=args['n_historical_events'], 
            n_time_steps=1000
            )
        all_data = np.hstack(sample[0])  # history, data = sample[0]

        for i in range(N):
            j = np.random.randint(0, all_data.shape[1] - args['n_historical_events'])
            history = all_data[:, j:j + args['n_historical_events']]
            if args['state_rep'] == 71: 
               initial_state = env.case.get_initial_state(history, adam_basket[i])
            else:
                raise NotImplementedError

            states, actions = pe.sample_from_policy2(env, model, obs_normalizer, initial_state=initial_state)
            states = np.array(states)
            actions = np.array(actions)

            a = pe.get_distrib(states, actions)

            temp_agents.append(a)

        agents.append(temp_agents)

    return agents

############################

class Result():
    def __init__(self, models):
        self.models = models

def get_key_from_path(path):
    temp = path.split('/')[-1]
    steps = int(temp.split('_')[0])  # training steps
    return steps

def get_distribs(all_states, all_actions):
    N = len(all_states)
    distribs = []
    for states, actions in zip(np.split(all_states, N), np.split(all_actions, N)):  # Loop over individuals
        distribs.append(pe.get_distrib(states, actions))
    return distribs

def save_result(result_obj, folder):
    with open(join(folder, 'result2.pkl'), 'wb') as f:
        pickle.dump(result_obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(folder):
    with open(join(folder, 'result2.pkl'), 'rb') as f:
        result = pickle.load(f)
    return result

def get_label_from_param_value(param_value):
    if param == 'adam_days':
        if param_value == 10:
            return '10 purchases'
        elif param_value == 20:
            return '20 purchases'
        elif param_value == 30:
            return '30 purchases'
    else:
        if param_value == 365:
            return '1 year'
        elif param_value == 730:
            return '2 years'
        elif param_value == 1095:
            return '3 years'
        elif param_value == 30:
            return '1 month'
        elif param_value == 60:
            return '2 months'
        elif param_value == 90:
            return '3 months'

def get_label_from_param(param):
    if param == 'n_historical_events':
        return 'Number of historical events'
    elif param == 'adam_days':
        return 'Number of Adam days'
    elif param == 'length_expert_TS':
        return 'Length of expert time-series'
    else:
        raise NotImplementedError

############################

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        save_data(path, sample_length=10000, n_new_customers=50, N=1)
    else:
        main()
