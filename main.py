from customer_behaviour.tools import dgm as dgm
from customer_behaviour.tools.generate_data import User 
import os
import argparse
import subprocess
import numpy as np

def main(case = 'discrete_events'):
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', default='gail', choices=['gail', 'airl'], type=str)
    parser.add_argument('--case', type=str, default=case)
    parser.add_argument('--n_experts', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=10000)
    args = parser.parse_args()

    env = find_env(args.case, args.n_experts)

    n_experts = 1

    demo_states = []
    demo_actions = []

    for i in range(n_experts):
        usr = User(time_steps = 128)
        states, actions = usr.generate_trajectory()
        demo_states.append(states)
        demo_actions.append(actions)

    PathOfDemonstrationNpzFile = os.getcwd() + '/expert_trajectories.npz'
    np.savez(PathOfDemonstrationNpzFile, states=np.array(demo_states, dtype=object),
             actions=np.array(demo_actions, dtype=object)) 
    command = ['python', 'customer_behaviour/algorithms/train_gym.py', args.algo, '--gpu', '-1', '--env', env, '--arch', 'FFSoftmax', '--steps', str(args.n_steps), '--load_demo', 'expert_trajectories.npz', '--update-interval', '128', '--entropy-coef' ,'0.01']
    s = subprocess.Popen(command)

def find_env(case, n_experts):
    if case == 'discrete_events':
        if n_experts == 1:
            env = 'discrete-buying-events-v0'

    if case == 'full_receipt':
        if n_experts == 1:
            env = ''

    return env


if __name__ == '__main__':
    main()
