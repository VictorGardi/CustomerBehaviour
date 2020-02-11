import subprocess

calls = []
"""GLÖM INTE ATT ÄNDRA GPU INNAN DU KÖR PÅ VM!!!!"""
call_1 = ['python', 'main.py', 'gail', '--gpu', '-1', '--n_training_episodes', '1', '--n_historical_events', '50', '--case', 'discrete_events', '--n_experts', '1', '--n_products', '1', '--episode_length', '200', '--length_expert_TS', '200', '--seed_expert', 'True', '--agent_seed', '0']
call_2 = ['python', 'main.py', 'gail', '--gpu', '-1', '--n_training_episodes', '2', '--n_historical_events', '50', '--case', 'discrete_events', '--n_experts', '1', '--n_products', '1', '--episode_length', '200', '--length_expert_TS', '200', '--seed_expert', 'True', '--agent_seed', '0']

calls.append(call_1, call_2)


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

    
