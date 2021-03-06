import subprocess
import argparse

def terminate_instance():
    call = ['sudo', 'poweroff']
    a = subprocess.Popen(call)

def get_params(par_value):
    if par_value == '365':
        n_training_episodes = '30000'  # -> n_tot_steps = 365 * 30000 = 10 950 000
        update_interval = '10950'  # steps
        eval_interval = '1095000'  # steps
    elif par_value == '730':
        n_training_episodes = '15000'  # -> n_tot_steps = 730 * 15000 = 10 950 000
        update_interval = '10950'  # steps
        eval_interval = '1095000'  # steps
    elif par_value == '1095':
        n_training_episodes = '10000'  # -> n_tot_steps = 1095 * 10000 = 10 950 000
        update_interval = '10950'  # steps
        eval_interval = '1095000'  # steps
    return n_training_episodes, update_interval, eval_interval

def get_call(case, par_value):
    if case == 'n_historical_events':
        call = ['python3', 'main.py', 'gail', '--gpu', '-1', '--case', 'discrete_events', '--n_experts', '10', '--state_rep', '71', '--n_historical_events', par_value, \
     '--episode_length', '1095', '--n_training_episodes', '10000', '--length_expert_TS', '1095', '--seed_expert', 'True', '--seed_agent', 'True', \
         '--n_demos_per_expert', '0', '--eval_episode_length', '10', '--update-interval', '10950', '--eval_interval', '1095000', '--D_layers', '64', '64', \
              '--G_layers', '64', '64', '--normalize_obs', 'False', '--eval-n-runs', '10', '--n_processes', '2', '--PAC_k', '1', '--gamma', '0', '--noise', '0', \
                  '--batchsize', '73', '--show_D_dummy', 'True', '--adam_days', '90', '--save_folder', case, '--save_results', 'True', '--novelnovel', 'True'] 

    elif case == 'episode_length':
        n_training_episodes, update_interval, eval_interval = get_params(par_value)

        call = ['python3', 'main.py', 'gail', '--gpu', '-1', '--case', 'discrete_events', '--n_experts', '10', '--state_rep', '71', '--n_historical_events', '30', \
     '--episode_length', par_value, '--n_training_episodes', n_training_episodes, '--length_expert_TS', '1095', '--seed_expert', 'True', '--seed_agent', 'True', \
         '--n_demos_per_expert', '0', '--eval_episode_length', '10', '--update-interval', update_interval, '--eval_interval', eval_interval, '--D_layers', '64', '64', \
              '--G_layers', '64', '64', '--normalize_obs', 'False', '--eval-n-runs', '10', '--n_processes', '2', '--PAC_k', '1', '--gamma', '0', '--noise', '0', \
                  '--batchsize', '73', '--show_D_dummy', 'True', '--adam_days', '90', '--save_folder', case, '--save_results', 'True', '--novelnovel', 'True'] 

    elif case == 'length_expert_TS':
        call = ['python3', 'main.py', 'gail', '--gpu', '-1', '--case', 'discrete_events', '--n_experts', '10', '--state_rep', '71', '--n_historical_events', '30', \
     '--episode_length', '1095', '--n_training_episodes', '10000', '--length_expert_TS', par_value, '--seed_expert', 'True', '--seed_agent', 'True', \
         '--n_demos_per_expert', '0', '--eval_episode_length', '10', '--update-interval', '10950', '--eval_interval', '1095000', '--D_layers', '64', '64', \
              '--G_layers', '64', '64', '--normalize_obs', 'False', '--eval-n-runs', '10', '--n_processes', '2', '--PAC_k', '1', '--gamma', '0', '--noise', '0', \
                  '--batchsize', '73', '--show_D_dummy', 'True', '--adam_days', '90', '--save_folder', case, '--save_results', 'True', '--novelnovel', 'True'] 

    elif case == 'adam_days':
        call = ['python3', 'main.py', 'gail', '--gpu', '-1', '--case', 'discrete_events', '--n_experts', '10', '--state_rep', '71', '--n_historical_events', '30', \
     '--episode_length', '1095', '--n_training_episodes', '10000', '--length_expert_TS', '1095', '--seed_expert', 'True', '--seed_agent', 'True', \
         '--n_demos_per_expert', '0', '--eval_episode_length', '10', '--update-interval', '10950', '--eval_interval', '1095000', '--D_layers', '64', '64', \
              '--G_layers', '64', '64', '--normalize_obs', 'False', '--eval-n-runs', '10', '--n_processes', '2', '--PAC_k', '1', '--gamma', '0', '--noise', '0', \
                  '--batchsize', '73', '--show_D_dummy', 'True', '--adam_days', par_value, '--save_folder', case, '--save_results', 'True', '--novelnovel', 'True'] 
    return call

def main(case='n_historical_events'):
    parser = argparse.ArgumentParser()
    parser.add_argument('case', default='n_historical_events', choices=['n_historical_events', 'episode_length', 'length_expert_TS', 'adam_days'], type=str)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--par_value', default=None, type=str)
    args = parser.parse_args()

    call = get_call(args.case, args.par_value)
    for i in range(args.n_runs):
        s = subprocess.Popen(call, stdout=subprocess.PIPE)
        output = s.stdout.readline()
    #terminate_instance()

if __name__ == "__main__":
    main()