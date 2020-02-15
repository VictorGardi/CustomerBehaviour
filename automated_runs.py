import subprocess

calls = []
"""GLÖM INTE ATT ÄNDRA GPU INNAN DU KÖR PÅ VM!!!!"""
call_1 = ['python', 'main.py', 'airl', '--gpu', '0','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
 '--n_historical_events', '20', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '3', \
  '--episode_length', '100', '--length_expert_TS', '100', '--seed_expert', 'True', '--agent_seed', '0']
call_2 = ['python', 'main.py', 'airl', '--gpu', '0','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
 '--n_historical_events', '25', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '3', \
  '--episode_length', '100', '--length_expert_TS', '100', '--seed_expert', 'True', '--agent_seed', '0']
call_3 = ['python', 'main.py', 'airl', '--gpu', '0','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
 '--n_historical_events', '30', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '3', \
  '--episode_length', '100', '--length_expert_TS', '100', '--seed_expert', 'True', '--agent_seed', '0']
call_4 = ['python', 'main.py', 'airl', '--gpu', '0','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
 '--n_historical_events', '35', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '3', \
  '--episode_length', '100', '--length_expert_TS', '100', '--seed_expert', 'True', '--agent_seed', '0']
call_5 = ['python', 'main.py', 'airl', '--gpu', '0','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
 '--n_historical_events', '45', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '3', \
  '--episode_length', '100', '--length_expert_TS', '100', '--seed_expert', 'True', '--agent_seed', '0']

calls.append(call_1)
calls.append(call_2)
calls.append(call_3)
calls.append(call_4)
calls.append(call_5)



def terminate_instance():
    call = ['sudo', 'poweroff']
    a = subprocess.Popen(call)
    
def main():
    for call in calls:
        s = subprocess.Popen(call, stdout=subprocess.PIPE)
        output = s.stdout.readline()
    terminate_instance()

        


if __name__ == "__main__":
    main()

    
