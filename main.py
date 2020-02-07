import numpy as np
import os
from customer_behaviour.tools import dgm as dgm
from customer_behaviour.tools.generate_data import User 
# import configparser as cp

config = cp.ConfigParser()
config.read('config.ini')

case = 

def main(case = 'discrete_events'):

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



if __name__ == '__main__':
    main()

