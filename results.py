import os
import sys 
from os.path import join
tools_path = join(os.getcwd(), 'customer_behaviour/tools')
sys.path.insert(1, tools_path)
import gym
import json
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
import customer_behaviour.tools.policy_evaluation as pe

from os.path import join
from collections import OrderedDict
from scipy.stats import wasserstein_distance as wd

############################

dir_path = 'n_historical_events'
param = dir_path
n_new_customers = 10
sample_length = 10000
resample = False
k = 1  # The number of closest experts
N = 1  # The number of reinitializations when predicting a new customer 

############################

# Utvärdera hur det ser ut på populationsnivå?

def main():
    if resample: input("Are you sure you want to resample all data?\nPress 'Enter' to continue...")

    data = load_data()

    fig1, ax1 = plt.subplots()
    ax1.set_xlabel(get_label_from_param(param))
    ax1.set_ylabel('Classification error (%)')

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel(get_label_from_param(param))
    ax2.set_ylabel('Absolute difference (EMD)')
     # Det skulle också vara intressant att jämföra med det absoluta avståndet till den expert vars dummyvariabel är satt till 1

    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('Training steps')
    ax3.set_ylabel('Classification error (%)')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Traning steps')
    ax4.set_ylabel('Absolute difference (EMD)') 
    # Inkludera variansen i agentens beteende 
    # Agenten beter sig bra på populationsnivå efter, säg, 2000 epsioder men påvisa att den inte har någon varians på individnivå
    # Kan exempelvis summera avståndet från agenten till samtliga experter och visa att avståndet minskar med tiden
    # Kan också summera inbördes avstånd mellan agentens olika beteenden och visa att det ökar med tiden

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (param_value, result) in enumerate(data.items()):

        for n_train_steps, t in result.models.items():

            if n_train_steps == 'final':
                _, abs_diffs, errors = t
                ax1.plot(param_value, np.mean(errors), 'o')
                ax2.plot(param_value, np.mean(abs_diffs), 'o')
            else:
                _, _, errors, emd_avg = t
                ax3.plot(n_train_steps, np.mean(errors), 'o', label=str(param_value), color=colors[i])
                ax4.plot(n_train_steps, emd_avg, 'o', label=str(param_value), color=colors[i])
    
    for ax in (ax3, ax4):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    plt.show()

def load_data():
    data = {}
    
    # Load data
    data_paths = [join(dir_path, x) for x in os.listdir(dir_path) if x.startswith('2020')]
    data_paths.sort()

    for path in data_paths:
        content = os.listdir(path)

        args = json.loads(open(join(path, 'args.txt'), 'r').read())

        if 'result.pkl' in content and not resample:
            result = load_result(path)

        else:
            models = {}

            model_paths = [d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('checkpoint')]
            model_paths.sort(key=get_key_from_path)

            for mp in model_paths:
                distribs, abs_diffs, errors = evaluate_on_new_customers(args, mp)
                emd_avg = evaluate_on_pop_level(args, mp)
                n_train_steps = get_key_from_path(mp)
                models[n_train_steps] = (distribs, abs_diffs, errors, emd_avg)

            final_model_path = next((d for d in [x[0] for x in os.walk(path)] if d.endswith('finish')), None)
            distribs, abs_diffs, errors = evaluate_on_new_customers(args, final_model_path)
            
            models['final'] = (distribs, abs_diffs, errors)

            result = Result(models)
            save_result(result, path)

        data[args[param]] = result

    return data

def evaluate_on_pop_level(args, model_path):
    env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False)

    # Sample customer data
    n_experts = args['n_experts']
    customer_trajectories = env.generate_expert_trajectories(
        out_dir=None, 
        n_demos_per_expert=1,
        n_experts=n_experts,
        n_expert_time_steps=sample_length
        )
    expert_states = np.array(customer_trajectories['states'])
    expert_actions = np.array(customer_trajectories['actions'])

    expert_distrib = pe.get_distrib(expert_states, expert_actions)

    agent_states = []
    agent_actions = []
    for i in range(n_experts):
        initial_state = random.choice(expert_states[i])
        states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
        agent_states.append(states)
        agent_actions.append(actions)
    agent_states = np.array(agent_states)
    agent_actions = np.array(agent_actions)

    agent_distrib = pe.get_distrib(agent_states, agent_actions)

    emd = wd(agent_distrib, expert_distrib)

    return emd

def evaluate_on_new_customers(args, model_path):
    env, model, obs_normalizer = pe.get_env_and_model(args, model_path, sample_length, only_env=False)

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

    expert_distribs = get_distribs(expert_states, expert_actions)
    new_distribs = get_distribs(new_states, new_actions)

    distribs = []
    abs_diffs = []
    errors = []

    for i, nb in enumerate(new_distribs):
        distances = [wd(nb, v) for v in expert_distribs]
        closest_experts = np.argsort(distances)[:k]
        dummy = closest_experts[0]

        temp_distribs = []
        temp_abs_diffs = []
        n_errors = 0

        for _ in range(N):
            initial_state = random.choice(new_states[i])
            initial_state[dummy] = 1

            states, actions = pe.sample_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            states = np.array(states)
            actions = np.array(actions)

            u = pe.get_distrib(states, actions)
            temp_distribs.append(u)
            temp_abs_diffs.append(wd(u, nb))

            distances = [wd(u, v) for v in expert_distribs]
            if np.argmin(distances) not in closest_experts:
                n_errors += 1

        distribs.append(temp_distribs)
        abs_diffs.append(temp_abs_diffs)
        errors.append(n_errors / N)

    return distribs, abs_diffs, errors

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
    main()