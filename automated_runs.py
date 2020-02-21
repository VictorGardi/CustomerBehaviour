import subprocess

calls = []
"""GLÖM INTE ATT ÄNDRA GPU INNAN DU KÖR PÅ VM!!!!"""
call_1 = ['python3', 'main.py', 'gail', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '32', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '128', '--seed_expert', 'True', '--agent_seed', '0']
call_2 = ['python3', 'main.py', 'gail', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '64', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '128', '--seed_expert', 'True', '--agent_seed', '0']
call_3 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '96', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '128', '--seed_expert', 'True', '--agent_seed', '0']
call_4 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '32', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '256', '--seed_expert', 'True', '--agent_seed', '0']
call_5 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '64', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '256', '--seed_expert', 'True', '--agent_seed', '0']
call_6 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '96', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '256', '--seed_expert', 'True', '--agent_seed', '0']
call_7 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '32', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '512', '--seed_expert', 'True', '--agent_seed', '0']
call_8 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '64', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '512', '--seed_expert', 'True', '--agent_seed', '0']
call_9 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '0','--n_demos_per_expert', '1', '--n_training_episodes', '10000', \
 '--n_historical_events', '96', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '2', \
  '--episode_length', '128', '--length_expert_TS', '512', '--seed_expert', 'True', '--agent_seed', '0']

calls.append(call_1)
calls.append(call_2)
#calls.append(call_3)
#calls.append(call_4)
#calls.append(call_5)



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

    
