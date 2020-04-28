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

############################

dir_path = 'n_historical_events'
param = dir_path
n_new_customers = 10
sample_length = 2000
resample = False
plot_only = True
update_classification = False
compare_features = False
k = 2  # The number of closest experts
N = 1  # The number of reinitializations when predicting a new customer 

############################

def main():
    if resample: input("Are you sure you want to resample all data?\nPress 'Enter' to continue...")

    data = load_data()

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(get_label_from_param(param))
    ax1.set_ylabel('Classification error (%)')

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(get_label_from_param(param))
    ax2.set_ylabel('Absolute difference (ED)') if compare_features else ax2.set_ylabel('Absolute difference (EMD)') 
    ax2.set_title('Comparison with %s new customers' % n_new_customers)
    # Det skulle också vara intressant att jämföra med det absoluta avståndet till den expert vars dummyvariabel är satt till 1

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('Training steps')
    ax3.set_ylabel('Classification error (%)')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Training steps')
    ax4.set_ylabel('Absolute difference (ED)') if compare_features else ax4.set_ylabel('Absolute difference (EMD)') 
    ax4.set_title('Comparison with experts in training data')
    # Inkludera variansen i agentens beteende 
    # Agenten beter sig bra på populationsnivå efter, säg, 2000 epsioder men påvisa att den inte har någon varians på individnivå
    # Kan exempelvis summera avståndet från agenten till samtliga experter och visa att avståndet minskar med tiden
    # Kan också summera inbördes avstånd mellan agentens olika beteenden och visa att det ökar med tiden

    param_values = list(data.keys())
    n_params = len(param_values)

    assert param_values
    step_values = list(data[param_values[0]][0].models.keys())[:-1]
    n_step_values = len(step_values)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    error_vs_param = [[] for i in range(n_params)]
    avg_dist_vs_param = [[] for i in range(n_params)]
    error_vs_step = [[[] for j in range(n_step_values)] for i in range(n_params)]
    avg_dist_vs_step = [[[] for j in range(n_step_values)] for i in range(n_params)]

    for param_value, results in data.items():
        i = param_values.index(param_value)

        for result in results:

            for n_train_steps, t in result.models.items():
                if n_train_steps == 'final':
                    _, abs_diffs, errors = t
                    error_vs_param[i].append(100 * np.mean(errors))
                    avg_dist_vs_param[i].append(np.mean(abs_diffs))
                else:
                    j = step_values.index(n_train_steps)
                    _, _, errors, avg_dist = t
                    error_vs_step[i][j].append(100 * np.mean(errors))
                    avg_dist_vs_step[i][j].append(avg_dist)

    error_vs_param = list(map(lambda x: np.mean(x), error_vs_param))
    avg_dist_vs_param = list(map(lambda x: np.mean(x), avg_dist_vs_param))
    ax1.plot(param_values, error_vs_param)
    ax2.plot(param_values, avg_dist_vs_param)

    for i in range(n_params):
        temp1 = list(map(lambda x: np.mean(x), error_vs_step[i]))
        temp2 = list(map(lambda x: np.mean(x), avg_dist_vs_step[i]))
        ax3.plot(step_values, temp1, label=str(param_values[i]), color=colors[i])
        ax4.plot(step_values, temp2, label=str(param_values[i]), color=colors[i])

    for ax in (ax3, ax4):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.show()

def save_data(path, sample_length, n_new_customers, compare_features):
    args = json.loads(open(join(path, 'args.txt'), 'r').read())

    n_experts = args['n_experts']

    # Sample customer data
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
    expert_states = np.array(customer_trajectories['states'][:n_experts])
    expert_actions = np.array(customer_trajectories['actions'][:n_experts])
    new_states = np.array(customer_trajectories['states'][n_experts:])
    new_actions = np.array(customer_trajectories['actions'][n_experts:])

    if compare_features:
        avg_expert = get_features(expert_actions, average=True)
        experts = get_features(expert_actions)
        new_customers = get_features(new_actions)
    else:
        avg_expert = pe.get_distrib(expert_states, expert_actions)
        experts = get_distribs(expert_states, expert_actions)
        new_customers = get_distribs(new_states, new_actions)

    models = {}

    model_paths = [d for d in [x[0] for x in os.walk(path)] if d.endswith('checkpoint')]
    model_paths.sort(key=get_key_from_path)

    for mp in model_paths:
        n_train_steps = get_key_from_path(mp)
        if int(n_train_steps) < 1000000: continue

        print('Collecting data from model: %s' % n_train_steps)

        agent, abs_diffs, errors = evaluate_on_new_customers(args, mp, experts, new_customers, compare_features)
        avg_dist = evaluate_on_pop_level(args, mp, avg_expert, compare_features)
        
        models[n_train_steps] = (agent, abs_diffs, errors, avg_dist)

    # final_model_path = next((d for d in [x[0] for x in os.walk(path)] if d.endswith('finish')), None)
    # agent, abs_diffs, errors = evaluate_on_new_customers(args, final_model_path, experts, new_customers, compare_features)
    
    # models['final'] = (agent, abs_diffs, errors)

    result = Result(models)
    save_result(result, path)

def load_data():
    data = {}
    
    # Load data
    data_paths = [join(dir_path, x) for x in os.listdir(dir_path) if x.startswith('2020')]
    data_paths.sort()

    for i, path in enumerate(data_paths):
        print('Processing folder {} of {}'.format(i + 1, len(data_paths)))

        content = os.listdir(path)

        args = json.loads(open(join(path, 'args.txt'), 'r').read())

        if not (plot_only and not update_classification) and i == 0:
            env = pe.get_env_and_model(args, '', sample_length, only_env=True)

            # Sample customer data
            n_experts = args['n_experts']
            customer_trajectories = env.generate_expert_trajectories(
                out_dir=None, 
                n_demos_per_expert=1,
                n_experts=n_experts+n_new_customers,
                n_expert_time_steps=sample_length
                )
            expert_states = np.array(customer_trajectories['states'][:n_experts])
            expert_actions = np.array(customer_trajectories['actions'][:n_experts])
            new_states = np.array(customer_trajectories['states'][n_experts:])
            new_actions = np.array(customer_trajectories['actions'][n_experts:])

            if compare_features:
                avg_expert = get_features(expert_actions, average=True)
                experts = get_features(expert_actions)
                new_customers = get_features(new_actions)
            else:
                avg_expert = pe.get_distrib(expert_states, expert_actions)
                experts = get_distribs(expert_states, expert_actions)
                new_customers = get_distribs(new_states, new_actions)

        if 'result.pkl' in content and not resample:
            print('Loading saved result')
            result = load_result(path)

            if update_classification:
                print('Updating classification with k = %d' % k)

                for n_train_steps, t in result.models.items():
                    agent = t[0]
                    errors = compare_with_new_customers(agent, experts, new_customers, compare_features)

                    l = list(t)
                    l[2] = errors
                    t = tuple(l)
                    result.models.update({n_train_steps: t})
        else:
            if plot_only: 
                print('Ignoring')
                continue

            print('Collecting result by sampling from model')
            
            models = {}

            model_paths = [d for d in [x[0] for x in os.walk(path)] if d.endswith('checkpoint')]
            model_paths.sort(key=get_key_from_path)

            for mp in model_paths:
                n_train_steps = get_key_from_path(mp)
                if n_train_steps < 1000000: continue

                agent, abs_diffs, errors = evaluate_on_new_customers(args, mp, experts, new_customers, compare_features)
                avg_dist = evaluate_on_pop_level(args, mp, avg_expert, compare_features)
                
                models[n_train_steps] = (agent, abs_diffs, errors, avg_dist)

            # final_model_path = next((d for d in [x[0] for x in os.walk(path)] if d.endswith('finish')), None)
            # agent, abs_diffs, errors = evaluate_on_new_customers(args, final_model_path, experts, new_customers, compare_features)
            
            # models['final'] = (agent, abs_diffs, errors)

            result = Result(models)
            save_result(result, path)

        if args[param] in data:
            data[args[param]].append(result)
        else:
            data[args[param]] = [result]

    return data

def evaluate_on_pop_level(args, model_path, avg_expert, compare_features):
    env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False)

    n_experts = args['n_experts']

    metric = ed if compare_features else wd

    agent_states = []
    agent_actions = []
    for i in range(n_experts):
        # Initialize agent with data from ith expert
        env.model.spawn_new_customer(i)
        sample = env.case.get_sample(
            n_demos_per_expert=1, 
            n_historical_events=args['n_historical_events'], 
            n_time_steps=1000
            )
        all_data = np.hstack(sample[0])  # history, data = sample[0]
        j = np.random.randint(0, all_data.shape[1] - args['n_historical_events'])
        history = all_data[:, j:j + args['n_historical_events']]
        initial_state = env.case.get_initial_state(history, i)

        states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
        agent_states.append(states)
        agent_actions.append(actions)
    agent_states = np.array(agent_states)
    agent_actions = np.array(agent_actions)

    avg_agent = get_features(agent_actions, average=True) if compare_features else pe.get_distrib(agent_states, agent_actions)

    distance = metric(avg_agent, avg_expert)

    return distance

def compare_with_new_customers(agents, experts, new_customers, compare_features):
    errors = []

    metric = ed if compare_features else wd

    for i, nc in enumerate(new_customers):
        distances = [metric(nc, e) for e in experts]
        closest_experts = np.argsort(distances)[:k]

        n_errors = 0

        for a in agents[i]:
            distances = [metric(a, e) for e in experts]
            if np.argmin(distances) not in closest_experts:
                n_errors += 1

        errors.append(n_errors / N)

    return errors

def evaluate_on_new_customers(args, model_path, experts, new_customers, compare_features):
    global k, N

    env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False)

    agents = []
    abs_diffs = []
    errors = []

    n_experts = args['n_experts']

    metric = ed if compare_features else wd

    for i, nc in enumerate(new_customers):
        distances = [metric(nc, e) for e in experts]
        closest_experts = np.argsort(distances)[:k]
        dummy = closest_experts[0]

        if args['state_rep'] == 71: adam_basket = np.random.permutation(env.case.adam_baskets[dummy])

        temp_agents = []
        temp_abs_diffs = []
        n_errors = 0

        seed = n_experts + i
        env.model.spawn_new_customer(seed)
        sample = env.case.get_sample(
            n_demos_per_expert=1, 
            n_historical_events=args['n_historical_events'], 
            n_time_steps=1000
            )
        all_data = np.hstack(sample[0])  # history, data = sample[0]

        for l in range(N):
            j = np.random.randint(0, all_data.shape[1] - args['n_historical_events'])
            history = all_data[:, j:j + args['n_historical_events']]
            if args['state_rep'] == 71:
                dummy = adam_basket[l]
            initial_state = env.case.get_initial_state(history, dummy)  # We set dummy to closest expert

            states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            states = np.array(states)
            actions = np.array(actions)

            a = get_features([actions]) if compare_features else pe.get_distrib(states, actions)

            temp_agents.append(a)
            temp_abs_diffs.append(metric(a, nc))

            distances = [metric(a, e) for e in experts]
            if np.argmin(distances) not in closest_experts:
                n_errors += 1

        agents.append(temp_agents)
        abs_diffs.append(temp_abs_diffs)
        errors.append(n_errors / N)

    return agents, abs_diffs, errors

############################

class Result():
    def __init__(self, models):
        self.models = models

def get_features(all_actions, average=False):
    features = []
    for actions in all_actions:
        avg, std = get_mean_std_purchase_freq(actions)
        # purchase_ratio = np.count_nonzero(actions) / len(actions)
        features.append([avg, std])
    if average:
        return np.mean(features, axis=0)
    else:
        return features

def get_mean_std_purchase_freq(actions):
    indices = np.argwhere(actions).reshape((-1,))
    assert indices.size > 0
    temp = np.diff(indices)
    return np.mean(temp), np.std(temp)

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
    with open(join(folder, 'result.pkl'), 'wb') as f:
        pickle.dump(result_obj, f, pickle.HIGHEST_PROTOCOL)

def load_result(folder):
    with open(join(folder, 'result.pkl'), 'rb') as f:
        result = pickle.load(f)
    return result

def get_label_from_param(param):
    if param == 'n_historical_events':
        return 'Number of historical events'
    else:
        raise NotImplementedError

############################

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        save_data(path, sample_length=10000, n_new_customers=50, compare_features=False)
    else:
        main()
