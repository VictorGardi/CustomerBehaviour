import sys
import os, re, json, random, itertools, time
from os.path import join
import pandas as pd
tools_path = join(os.getcwd(), 'customer_behaviour/tools')
sys.path.insert(1, tools_path)
import policy_evaluation as pe
from result import Result
from tools import save_plt_as_eps, save_plt_as_png
import gym
import custom_gym
import chainer
import chainerrl
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dgm import DGM
import random

from chainerrl.misc.batch_states import batch_states
def sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=None):
    xp = np
    phi = lambda x: x

    probs = []
    actions = []

    obs = env.reset().astype('float32')  # Initial state

    if initial_state is not None:
        env.state = initial_state
        obs = np.array(initial_state).astype('float32')

    done = False
    while not done:
        b_state = batch_states([obs], xp, phi)
        
        if obs_normalizer:
            b_state = obs_normalizer(b_state, update=False)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib, _ = model(b_state)
            action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]

        probs.append(action_distrib.all_prob.data[0][-1])
        actions.append(action)
        new_obs, _, done, _ = env.step(action)
        obs = new_obs.astype('float32')

    return probs, actions

def get_probs_for_buys(env, model, obs_normalizer, args, model_dir_path, expert=1, sample_length=5000, model_path=None, init='ones', dummi=True, rand_seed=None):
    n_input_neurons = model.pi.model.in_size
    dummy = [0]*args['n_experts']
    if dummi:
        dummy[expert] = 1
    
    if init == 'expert':
        expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
        expert_states = expert_trajectories['states']
        expert_actions = expert_trajectories['actions']
        e_actions = expert_actions[expert]
        e_states = expert_states[expert]
        index = np.random.randint(len(e_states))
        initial_state = e_states[index].tolist()
    elif init == 'ones':
        initial_state = dummy + [1]*(n_input_neurons-args['n_experts'])
    elif init == 'zeros':
        initial_state = dummy + [0]*(n_input_neurons-args['n_experts'])
        
    elif init == 'rand':
        from customer_behaviour.tools.dgm import DGM
        dgm = DGM()
        if rand_seed:
            dgm.spawn_new_customer(rand_seed)
            rand = rand_seed
        else:
            rand = np.random.randint(11,1000)
            dgm.spawn_new_customer(rand)
        #dgm.spawn_new_customer(12)
        sample = dgm.sample((n_input_neurons-args['n_experts']))
        sample = np.sum(sample, axis=0)
        initial_state = np.concatenate((dummy,sample))
        probs, actions = sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=initial_state)
        return actions, rand
    probs, actions = sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=initial_state)
    return actions

def get_purchase_ratio(sequence, gamma=None):
    if len(sequence) == 0:
        return np.count_nonzero(sequence)/1
    else: 
        if gamma:
            sequence=sequence[int(gamma*len(sequence)):]
        return np.count_nonzero(sequence)/len(sequence)

def get_expert_actions(expert, sample_length):
    dgm = DGM()
    dgm.spawn_new_customer(expert)
    sample = dgm.sample(sample_length)
    sample = np.sum(sample, axis=0)
    return sample

def main():
    dir_path = 'ozzy_results/dummy/2020-05-08_14-50-32'
    experts = [2, 7, 8]
    model_path = None # join(os.getcwd(), dir_path, '12288000_checkpoint', 'model.npz')
    n_runs = 10
    sample_length = 5000
    cutoff = 300
    normalize = True
    n_demos_per_expert = 1
    n_last_days = 7
    max_n_purchases_per_n_last_days = 2
    show_info = True
    show_plots = False
    save_plots = True
    cluster_comparison = False
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    data = []
    
    #expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    #expert_states = expert_trajectories['states']
    #expert_actions = expert_trajectories['actions']
    #temp = []
    #for e in range(len(expert_actions)):
    #    temp.append([get_purchase_ratio(expert_actions[e][:i], gamma=0.3) for i in range(len(expert_actions[e]))])
    #temp = np.array(temp)
    #mean_purchase_ratio = np.mean(temp, axis=0)
    for expert in experts:
        if expert == 2:
            lab = 'Two'
        elif expert == 7:
            lab = 'Five'
        elif expert == 8:
            lab = 'Nine'
        e_actions = get_expert_actions(expert, sample_length*n_runs)
        action_chunks = [e_actions[x:x+sample_length] for x in range(0, len(e_actions), sample_length)]
        for chunk in action_chunks:
            agent_actions = get_probs_for_buys(env, model, obs_normalizer, args, model_dir_path, expert=expert, init='zeros', dummi=True)
            agt_purchase_ratio = [get_purchase_ratio(agent_actions[:i], gamma=0.7) for i in range(len(agent_actions))]
            exp_purchase_ratio = [get_purchase_ratio(chunk[:i], gamma=0.7) for i in range(len(chunk))]

            for i, (avg_agent, avg_expert) in enumerate(zip(agt_purchase_ratio, exp_purchase_ratio)):
                if i > cutoff:
                    if i % 10 == 0:
                        data.append([i, avg_agent, 'Agent', lab])
                        data.append([i, avg_expert, 'Expert',  lab])
        
    df = pd.DataFrame(data, columns=['Days', 'Purchase ratio', 'Data', 'Customer'])
    sns.set(style='darkgrid')
    
    g = sns.relplot(x='Days', y='Purchase ratio', hue='Customer', ci=95, kind='line', data=df, facet_kws={'legend_out': False}, style='Data')
    ax = g.axes[0][0]

    handles, labels = ax.get_legend_handles_labels()
    labels[1] = "3"
    labels[2] = "8"
    labels[3] = "9"

    ax.legend(handles, labels)

    #labels2, handles2 = zip(*sorted(zip(labels[1:], handles[1:]), key=lambda t: int(t[0].split(' ')[0])))
    #labels2 = list(labels2)
    #handles2 = list(handles2)
    #labels2.insert(0, get_label_from_param(param))
    #handles2.insert(0, handles[0])
    #ax.legend(handles2, labels2)
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles=handles[1:], labels=labels[1:])
    plt.show()
    g.fig.savefig(os.getcwd() + '/zeros_dummy_10.pdf', format='pdf')

def main2(): #
    dir_path = 'ozzy_results/dummy/2020-05-08_14-50-32'
    #experts = [2, 7, 8]
    rand_list = [33, 44]
    model_path = None # join(os.getcwd(), dir_path, '12288000_checkpoint', 'model.npz')
    n_runs = 10
    sample_length = 5000
    cutoff = 300
    normalize = True
    n_demos_per_expert = 1
    n_last_days = 7
    max_n_purchases_per_n_last_days = 2
    show_info = True
    show_plots = False
    save_plots = True
    cluster_comparison = False
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    data = []
    
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    temp = []
    for e in range(len(expert_actions)):
       temp.append([get_purchase_ratio(expert_actions[e][:i], gamma=0.3) for i in range(len(expert_actions[e]))])
    temp = np.array(temp)
    mean_purchase_ratio = np.mean(temp, axis=0)
    for i, avg in enumerate(mean_purchase_ratio):
        if i > cutoff:
            if i % 10 == 0:
                data.append([i, avg, 'Average expert', 'Ground truth'])
    dgm = DGM()
    for idx, rand in enumerate(rand_list):
        dgm.spawn_new_customer(rand)
        #dgm.spawn_new_customer(12)
        sample = dgm.sample(sample_length*n_runs)
        sample = np.sum(sample, axis=0)
        sample_chunks = [sample[x:x+sample_length] for x in range(0, len(sample), sample_length)]

        for chunk in sample_chunks:
            agent_actions, rand_list = get_probs_for_buys(env, model, obs_normalizer, args, model_dir_path, expert=None, init='rand', dummi=False, rand_seed = rand)
            agt_purchase_ratio = [get_purchase_ratio(agent_actions[:i], gamma=0.7) for i in range(len(agent_actions))]
            
            sample_purchase_ratio = [get_purchase_ratio(chunk[:i], gamma=0.7) for i in range(len(chunk))]

            for i, (avg_agent, avg_sample) in enumerate(zip(agt_purchase_ratio, sample_purchase_ratio)):
                if i > cutoff:
                    if i % 10 == 0:
                        data.append([i, avg_sample, 'New customer ' + str(idx + 1), 'Ground truth'])
                        data.append([i, avg_agent, 'New customer ' + str(idx + 1), 'Agent'])
                        
        
    df = pd.DataFrame(data, columns=['Day', 'Purchase ratio', 'Customer', 'Data'])
    sns.set(style='darkgrid')
    
    g = sns.relplot(x='Day', y='Purchase ratio', hue='Customer', ci=95, kind='line', data=df, facet_kws={'legend_out': True}, style='Data')
    ax = g.axes[0][0]

    handles, labels = ax.get_legend_handles_labels()

    #labels[1] = "3"
    #labels[2] = "8"
    #labels[3] = "9"

    #ax.legend(handles, labels)

    #labels2, handles2 = zip(*sorted(zip(labels[1:], handles[1:]), key=lambda t: int(t[0].split(' ')[0])))
    #labels2 = list(labels2)
    #handles2 = list(handles2)
    #labels2.insert(0, get_label_from_param(param))
    #handles2.insert(0, handles[0])
    #ax.legend(handles2, labels2)
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles=handles[1:], labels=labels[1:])
    plt.show()
    g.fig.savefig(os.getcwd() + '/rand.pdf', format='pdf')

def main_new():
    import statistics
    dir_path = 'ozzy_results/dummy/2020-05-08_14-50-32'
    #experts = [2, 7, 8]
    #rand_list = random.sample(range(11, 100), 5)
    rand_list = [18, 33, 55, 70, 94]
    model_path = None # join(os.getcwd(), dir_path, '12288000_checkpoint', 'model.npz')
    n_runs = 10
    sample_length = 5000
    cutoff = 300
    normalize = True
    n_demos_per_expert = 1
    n_last_days = 7
    max_n_purchases_per_n_last_days = 2
    show_info = True
    show_plots = False
    save_plots = True
    cluster_comparison = False
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    data = []
    
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    temp = []
    for e in range(len(expert_actions)):
       temp.append(get_purchase_ratio(expert_actions[e]))
    mean_purchase_ratio = statistics.mean(temp)

    dgm = DGM()
    n_input_neurons = model.pi.model.in_size
    dummy = [0]*args['n_experts']
    for idx, rand in enumerate(rand_list):
        agt_res = []
        sample_res = []
        dgm.spawn_new_customer(rand)
        #dgm.spawn_new_customer(12)
        sample = dgm.sample(sample_length*n_runs)
        sample = np.sum(sample, axis=0)
        sample_chunks = [sample[x:x+sample_length] for x in range(0, len(sample), sample_length)]

        dgm.spawn_new_customer(rand)

        sample = dgm.sample((n_input_neurons-args['n_experts'])*n_runs)
        sample = np.sum(sample, axis=0)
        initial_chunks =  [sample[x:x+(n_input_neurons-args['n_experts'])] for x in range(0, len(sample), (n_input_neurons-args['n_experts']))]

        for chunk, initial_chunk in zip(sample_chunks, initial_chunks):
            initial_state = np.concatenate((dummy,initial_chunk))
            _, agent_actions = sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            agt_res.append(get_purchase_ratio(agent_actions))
            
            sample_res.append(get_purchase_ratio(chunk))
        print(mean_purchase_ratio)
        print('mean agent ' + str(rand))
        print(statistics.mean(agt_res))
        print('std')
        print(statistics.stdev(agt_res))
        print('mean expert ' + str(rand))
        print(statistics.mean(sample_res))
        print('std')
        print(statistics.stdev(sample_res))


def main_new1():
    import statistics
    dir_path = 'ozzy_results/dummy/2020-05-08_14-50-32'
    #experts = [2, 7, 8]
    #rand_list = random.sample(range(11, 100), 5)
    ls = [1, 2, 3, 4, 5]
    model_path = None # join(os.getcwd(), dir_path, '12288000_checkpoint', 'model.npz')
    n_runs = 10
    sample_length = 5000
    cutoff = 300
    normalize = True
    n_demos_per_expert = 1
    n_last_days = 7
    max_n_purchases_per_n_last_days = 2
    show_info = True
    show_plots = False
    save_plots = True
    cluster_comparison = False
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    data = []
    
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length*n_runs)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    temp = []
    for e in range(len(expert_actions)):
      temp.append(get_purchase_ratio(expert_actions[e]))
    mean_purchase_ratio = statistics.mean(temp)

    #dgm = DGM()
    n_input_neurons = model.pi.model.in_size
    dummy = [0]*args['n_experts']
    for exp in ls:
        agt_res = []
        sample_res = []
        #dgm.spawn_new_customer(exp)
        #sample = dgm.sample(sample_length*n_runs)
        #sample = np.sum(sample, axis=0)
        #sample_chunks = [sample[x:x+sample_length] for x in range(0, len(sample), sample_length)]
        sample = expert_actions[exp]
        sample_chunks = [sample[x:x+sample_length] for x in range(0, len(sample), sample_length)]

        #dgm.spawn_new_customer(exp)

        #sample = dgm.sample((n_input_neurons-args['n_experts'])*n_runs)
        #sample = np.sum(sample, axis=0)
        #initial_chunks =  [sample[x:x+(n_input_neurons-args['n_experts'])] for x in range(0, len(sample), (n_input_neurons-args['n_experts']))]

        for chunk, initial_chunk in zip(sample_chunks, expert_states[exp]):
            
            initial_state = np.concatenate((dummy,initial_chunk.tolist()))
            _, agent_actions = sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            agt_res.append(get_purchase_ratio(agent_actions))
            sample_res.append(get_purchase_ratio(chunk))

        print('mean agent ' + str(exp))
        print(statistics.mean(agt_res))
        print('std')
        print(statistics.stdev(agt_res))
        print('mean expert ' + str(exp))
        print(statistics.mean(sample_res))
        print('std')
        print(statistics.stdev(sample_res))      

def main_new2():
    import statistics
    dir_path = 'ozzy_results/dummy/2020-05-08_14-50-32'
    #experts = [2, 7, 8]
    #rand_list = random.sample(range(11, 100), 5)
    ls = [0, 1, 2, 3, 4]
    model_path = None # join(os.getcwd(), dir_path, '12288000_checkpoint', 'model.npz')
    n_runs = 10
    sample_length = 5000
    cutoff = 300
    normalize = True
    n_demos_per_expert = 1
    n_last_days = 7
    max_n_purchases_per_n_last_days = 2
    show_info = True
    show_plots = False
    save_plots = True
    cluster_comparison = False
    args_path = join(dir_path, 'args.txt')
    args = json.loads(open(args_path, 'r').read())

    info = pe.get_info(args)

    # Get path of model 
    model_dir_path = next((d for d in [x[0] for x in os.walk(dir_path)] if d.endswith('finish')), None)

    os.makedirs(join(dir_path, 'figs'), exist_ok=True)

    ending_eps = '_normalize.eps' if normalize else '.eps'
    ending_png = '_normalize.png' if normalize else '.png'
    env, model, obs_normalizer = pe.get_env_and_model(args, model_dir_path, sample_length, model_path=model_path)

    data = []
    
    expert_trajectories = env.generate_expert_trajectories(out_dir=None, n_demos_per_expert=1, n_expert_time_steps=sample_length)
    expert_states = expert_trajectories['states']
    expert_actions = expert_trajectories['actions']
    temp = []
    for e in range(len(expert_actions)):
      temp.append(get_purchase_ratio(expert_actions[e]))
    mean_purchase_ratio = statistics.mean(temp)
    print(mean_purchase_ratio)

    dgm = DGM()
    n_input_neurons = model.pi.model.in_size
    
    for exp in ls:
        agt_res = []
        sample_res = []
        dgm.spawn_new_customer(exp)
        sample = dgm.sample(sample_length*n_runs)
        sample = np.sum(sample, axis=0)
        sample_chunks = [sample[x:x+sample_length] for x in range(0, len(sample), sample_length)]
        dummy = [0]*args['n_experts']
        dummy[exp] = 1

        #dgm.spawn_new_customer(exp)

        #sample = dgm.sample((n_input_neurons-args['n_experts'])*n_runs)
        #sample = np.sum(sample, axis=0)
        #initial_chunks =  [sample[x:x+(n_input_neurons-args['n_experts'])] for x in range(0, len(sample), (n_input_neurons-args['n_experts']))]

        for chunk in sample_chunks:
            initial_chunk = [0]*(n_input_neurons-args['n_experts'])
            initial_state = np.concatenate((dummy,initial_chunk))
            _, agent_actions = sample_probs_and_actions_from_policy(env, model, obs_normalizer, initial_state=initial_state)
            agt_res.append(get_purchase_ratio(agent_actions))
            sample_res.append(get_purchase_ratio(chunk))

        print('mean agent ' + str(exp))
        print(statistics.mean(agt_res))
        print('std')
        print(statistics.stdev(agt_res))
        print('mean expert ' + str(exp))
        print(statistics.mean(sample_res))
        print('std')
        print(statistics.stdev(sample_res))  
    

if __name__ == '__main__':
    #main() # zeros, dummy=True
    main2() #rand, dummy=False
    #main3() #expertstate, dummy=false
    #main_new() # rand, dummy = false
    #main_new1() # expert_state, dummy=false
    #main_new2() # zeros, dummy=true
