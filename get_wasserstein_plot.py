import os
import gym
import custom_gym
import chainer
import itertools
import chainerrl
import numpy
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from main import A3CFFSoftmax
from math import floor
from chainerrl.misc.batch_states import batch_states
from scipy.stats import chisquare, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from customer_behaviour.tools.tools import save_plt_as_eps, save_plt_as_png, sample_from_policy,\
    get_cond_val_states, reduce_dimensionality, sort_possible_val_states, get_counts

policy_directories = ['ozzy_results/airl/discrete_events/1_expert(s)/case_21/2020-03-15_21-07-57', \
    'ozzy_results/airl/discrete_events/1_expert(s)/case_21/2020-03-15_07-49-19', \
        'ozzy_results/airl/discrete_events/1_expert(s)/case_21/2020-03-15_07-47-01', \
            'ozzy_results/airl/discrete_events/1_expert(s)/case_21/2020-03-15_07-44-24', \
                'ozzy_results/airl/discrete_events/1_expert(s)/case_21/2020-03-15_21-08-30']

sample_length = 10000
normalize_counts = True
# n_experts = 1
n_last_days = 7
max_n_purchases_per_n_last_days = 2
alpha = 0.01  # significance level
parameter_to_plot = 'n_historical_events'

def main():
    saved_w_purchase = []
    saved_w_no_purchase = []
    saved_parameter = []
    for directory in policy_directories:
        # Load parameters and settings
        args_path = join(directory, 'args.txt')
        args = json.loads(open(args_path, 'r').read())
        saved_parameter.append(args[parameter_to_plot])

        # Create environment
        env = gym.make('discrete-buying-events-v0')
        try:
            env.initialize_environment(
                case=args['state_rep'], 
                n_historical_events=args['n_historical_events'], 
                episode_length=sample_length,  # length of the agent's sample
                n_experts=args['n_experts'],
                n_demos_per_expert=1,
                n_expert_time_steps=sample_length,  # length of expert's sample
                seed_agent=args['seed_agent'],
                seed_expert=args['seed_expert']
                )
        except KeyError:
            # seed_agent was not an argument 
            env.initialize_environment(
                case=args['state_rep'], 
                n_historical_events=args['n_historical_events'], 
                episode_length=sample_length,  # length of the agent's sample
                n_experts=args['n_experts'],
                n_demos_per_expert=1,
                n_expert_time_steps=sample_length,  # length of expert's sample
                seed_agent=True,
                seed_expert=args['seed_expert']
                )

        # Initialize model and observation normalizer
        model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=(64, 64))  # Assuming state = [historical purchases]
        obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], clip_threshold=5)

        # Load model and observation normalizer
        model_directory = next((d for d in [x[0] for x in os.walk(directory)] if d.endswith('finish') == 1), None)
        chainer.serializers.load_npz(join(model_directory, 'model.npz'), model)
        chainer.serializers.load_npz(join(model_directory, 'obs_normalizer.npz'), obs_normalizer)

        # Sample agent data
        agent_states = []
        agent_actions = []
        for i in range(args['n_experts']):
            temp_states, temp_actions = sample_from_policy(env, model, obs_normalizer)
            agent_states.append(temp_states)
            agent_actions.append(temp_actions)

        # Sample expert data
        trajectories = env.generate_expert_trajectories(out_dir='.', eval=False)
        os.remove('expert_trajectories.npz')  # Remove file containing expert trajectories
        expert_states = trajectories['states']
        expert_actions = trajectories['actions']

        # Get conditional validation states
        agent_purchase, agent_no_purchase = get_cond_val_states(agent_states, agent_actions, n_last_days)
        expert_purchase, expert_no_purchase = get_cond_val_states(expert_states, expert_actions, n_last_days)

        # Reduce the dimensionality by treating all validation states with more than x purchases as one single state
        expert_purchase = reduce_dimensionality(expert_purchase, max_n_purchases_per_n_last_days)
        expert_no_purchase = reduce_dimensionality(expert_no_purchase, max_n_purchases_per_n_last_days)
        agent_purchase = reduce_dimensionality(agent_purchase, max_n_purchases_per_n_last_days)
        agent_no_purchase = reduce_dimensionality(agent_no_purchase, max_n_purchases_per_n_last_days)

        # Get possible validation states
        possible_val_states = [list(x) for x in itertools.product([0, 1], repeat=n_last_days)]
        possible_val_states = reduce_dimensionality(possible_val_states, max_n_purchases_per_n_last_days, True)
        possible_val_states = sort_possible_val_states(possible_val_states)

        # Get counts
        expert_counts_purchase = get_counts(expert_purchase, possible_val_states, normalize=normalize_counts)
        expert_counts_no_purchase = get_counts(expert_no_purchase, possible_val_states, normalize=normalize_counts)
        agent_counts_purchase = get_counts(agent_purchase, possible_val_states, normalize=normalize_counts)
        agent_counts_no_purchase = get_counts(agent_no_purchase, possible_val_states, normalize=normalize_counts)

        # Calculate Wasserstein distances
        if normalize_counts:
            wd_purchase = wasserstein_distance(expert_counts_purchase, agent_counts_purchase)
            wd_no_purchase = wasserstein_distance(expert_counts_no_purchase, agent_counts_no_purchase)
        else:
            u = np.array(expert_counts_purchase)
            v = np.array(agent_counts_purchase)
            wd_purchase = wasserstein_distance(u / np.sum(u), v / np.sum(v))
            u = np.array(expert_counts_no_purchase)
            v = np.array(agent_counts_no_purchase)
            wd_no_purchase = wasserstein_distance(u / np.sum(u), v / np.sum(v))
        
        saved_w_purchase.append(wd_purchase)
        saved_w_no_purchase.append(wd_no_purchase)
    
    plt.figure(1)
    plt.scatter(saved_parameter, saved_w_purchase)
    plt.xlabel(parameter_to_plot)
    plt.ylabel('Wasserstein distance')
    plt.title('Purchase')


    plt.figure(2)
    plt.scatter(saved_parameter, saved_w_no_purchase)
    plt.xlabel(parameter_to_plot)
    plt.ylabel('Wasserstein distance')
    plt.title('No Purchase')

    plt.show()


if __name__ == '__main__':
    main()