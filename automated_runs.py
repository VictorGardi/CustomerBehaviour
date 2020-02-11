import subprocess

calls = []
"""GLÖM INTE ATT ÄNDRA GPU INNAN DU KÖR PÅ VM!!!!"""
call_1 = ['python', 'main.py', 'gail', '--gpu', '0', '--n_training_episodes', '10000', '--n_historical_events', '30', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '1', '--episode_length', '365', '--length_expert_TS', '365', '--seed_expert', 'True', '--agent_seed', '0']
call_2 = ['python', 'main.py', 'gail', '--gpu', '0', '--n_training_episodes', '10000', '--n_historical_events', '60', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '1', '--episode_length', '365', '--length_expert_TS', '365', '--seed_expert', 'True', '--agent_seed', '0']
call_3 = ['python', 'main.py', 'gail', '--gpu', '0', '--n_training_episodes', '10000', '--n_historical_events', '90', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '1', '--episode_length', '365', '--length_expert_TS', '365', '--seed_expert', 'True', '--agent_seed', '0']
call_4 = ['python', 'main.py', 'gail', '--gpu', '0', '--n_training_episodes', '10000', '--n_historical_events', '120', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '1', '--episode_length', '365', '--length_expert_TS', '365', '--seed_expert', 'True', '--agent_seed', '0']
call_5 = ['python', 'main.py', 'gail', '--gpu', '0', '--n_training_episodes', '10000', '--n_historical_events', '150', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '1', '--episode_length', '365', '--length_expert_TS', '365', '--seed_expert', 'True', '--agent_seed', '0']


calls.extend([[call_1], [call_2]])


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

    
