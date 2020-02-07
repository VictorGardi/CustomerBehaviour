from customer_behaviour.tools import dgm as dgm
from customer_behaviour.tools.generate_data import User 
import os
import argparse
import numpy as np

# import configparser as cp

config = cp.ConfigParser()
config.read('config.ini')


def main(case = 'discrete_events'):
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', default='ppo', choices=['gail', 'airl'], type=str)
    parser.add_argument('--case', type=str, default=case)
    args = parser.parse_args()

    n_experts = config[case]['N_EXPERTS']

    n_experts = 1

    demo_states = []
    demo_actions = []

    for i in range(n_experts):
        usr = User(time_steps = 128)
        states, actions = usr.generate_trajectory()
        demo_states.append(states)
        demo_actions.append(actions)

    print(demo_states)
    print(demo_actions)

    np.savez(os.getcwd() + '/expert_trajectories.npz', states=np.array(demo_states, dtype=object),
             actions=np.array(demo_actions, dtype=object)) 

    
    os.system()


if __name__ == '__main__':
    main()

