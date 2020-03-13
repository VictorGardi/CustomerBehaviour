import subprocess

calls = []
"""GLÖM INTE ATT ÄNDRA GPU INNAN DU KÖR PÅ VM!!!!"""
call_1 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '-1','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
'--n_historical_events', '30', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '21', '--episode_length', '1024',\
'--length_expert_TS', '1024', '--seed_expert', 'True', '--seed_agent', 'True', '--eval_episode_length', '2048',\
'--update-interval', '4096', '--eval_interval', '10240']
call_2 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '-1','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
'--n_historical_events', '60', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '21', '--episode_length', '1024',\
'--length_expert_TS', '1024', '--seed_expert', 'True', '--seed_agent', 'True', '--eval_episode_length', '2048',\
'--update-interval', '4096', '--eval_interval', '10240']
call_3 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '-1','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
'--n_historical_events', '90', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '21', '--episode_length', '1024',\
'--length_expert_TS', '1024', '--seed_expert', 'True', '--seed_agent', 'True', '--eval_episode_length', '2048',\
'--update-interval', '4096', '--eval_interval', '10240']
call_4 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '-1','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
'--n_historical_events', '120', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '21', '--episode_length', '1024',\
'--length_expert_TS', '1024', '--seed_expert', 'True', '--seed_agent', 'True', '--eval_episode_length', '2048',\
'--update-interval', '4096', '--eval_interval', '10240']
call_5 = ['python3', 'main.py', 'airl', '--update-interval', '2048', '--gpu', '-1','--n_demos_per_expert', '20', '--n_training_episodes', '10000', \
'--n_historical_events', '150', '--case', 'discrete_events', '--n_experts', '1', '--state_rep', '21', '--episode_length', '1024',\
'--length_expert_TS', '1024', '--seed_expert', 'True', '--seed_agent', 'True', '--eval_episode_length', '2048',\
'--update-interval', '4096', '--eval_interval', '10240']
 

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
    #terminate_instance()

        


if __name__ == "__main__":
    main()

    
