import os
import json
import numpy as np
import matplotlib.pyplot as plt
import customer_behaviour.tools.policy_evaluation as pe
from os.path import join
from customer_behaviour.tools.tools import save_plt_as_png

dir_path = 'results_anton/2020-03-18_14-01-33'

normalize = True
n_last_days = 7
max_n_purchases_per_n_last_days = 2
show_info = True
save_plots = True

def main():
    # Load arguments
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)
    n_sampled_days = args['eval_episode_length'] * args['n_experts']

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)
    ending_png = '_normalize.png' if normalize else '.png'

    # Load expert data
    expert_data_path = join(dir_path, 'eval_expert_trajectories.npz')
    expert_states, expert_actions = load_data(expert_data_path)

    expert_purchase, expert_no_purchase, expert_n_shopping_days = pe.get_cond_distribs(
        expert_states, 
        expert_actions, 
        n_last_days, 
        max_n_purchases_per_n_last_days, 
        normalize
        )

    # Load agent data
    agent_data_dir_path = join(dir_path, 'states_actions')
    agent_data_paths = os.listdir(agent_data_dir_path)
    agent_data_paths = [join(agent_data_dir_path, x) for x in agent_data_paths if x.endswith('npz')]
    agent_data_paths.sort(key=get_key_from_path)

    training_purchase = []
    training_no_purcahse = []

    for adp in agent_data_paths:
        n_updates = get_key_from_path(adp)
        agent_states, agent_actions = load_data(adp)

        agent_purchase, agent_no_purchase, agent_n_shopping_days = pe.get_cond_distribs(
        agent_states, 
            agent_actions, 
            n_last_days, 
            max_n_purchases_per_n_last_days, 
            normalize
            )

        # Calculate Wasserstein distances
        wd_purchase = pe.get_wd(expert_purchase, agent_purchase, normalize)
        wd_no_purchase = pe.get_wd(expert_no_purchase, agent_no_purchase, normalize)

        training_purchase.append(wd_purchase)
        training_no_purcahse.append(wd_no_purchase)
        
        '''
        agent_shopping_ratio = format(agent_n_shopping_days / n_sampled_days, '.3f')
        expert_shopping_ratio = format(expert_n_shopping_days / n_sampled_days, '.3f')
        expert_str = 'Expert (p.r.: ' + str(expert_shopping_ratio) + ')'
        agent_str = 'Agent (p.r.: ' + str(agent_shopping_ratio) + ')'

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Number of Adam steps: %d' % n_updates)

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
        if save_plots: save_plt_as_png(fig, path=join(dir_path, 'figs', str(n_updates) + ending_png))

        plt.show()
        '''

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.20)
    
    training_steps = [get_key_from_path(x) for x in agent_data_paths]
    
    ax1.plot(training_steps, training_purchase)
    ax1.set_xlabel('Number of training steps')
    ax1.xaxis.set_tick_params(rotation=90)
    ax1.set_ylabel('EMD')
    ax1.set_title('Purchase')
    
    ax2.plot(training_steps, training_no_purcahse)
    ax2.set_xlabel('Number of training steps')
    ax2.xaxis.set_tick_params(rotation=90)
    ax2.set_ylabel('EMD')
    ax2.set_title('No purchase')

    plt.show()

def get_key_from_path(path):
    temp = path.split('_')
    return int(temp[-1].split('.')[0])

def load_data(file):
    data = np.load(file, allow_pickle=True)
    assert sorted(data.files) == sorted(['states', 'actions'])
    states = data['states']
    actions = data['actions']
    return states, actions

if __name__ == '__main__':
    main()