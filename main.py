from customer_behaviour.tools import dgm as dgm
from customer_behaviour.tools.generate_data import User 
from customer_behaviour.algorithms.train_gym import run_algo
import os
import datetime
import argparse
import subprocess
import numpy as np
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', default='gail', choices=['gail', 'airl'], type=str)
    parser.add_argument('--case', type=str, default='discrete_events')
    parser.add_argument('--n_experts', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=10000)
    

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--env', type=str, default='discrete-buying-events-v0')
    parser.add_argument('--arch', type=str, default='FFGaussian',
                        choices=('FFSoftmax', 'FFMellowmax',
                                 'FFGaussian'))
    parser.add_argument('--bound-mean', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='/results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 6)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--load_demo', type=str, default=os.getcwd() + '/expert_trajectories.npz')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')

    parser.add_argument('--update-interval', type=int, default=2048)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.0)
    args = parser.parse_args()
    env = find_env(args.case, args.n_experts)
    outdir = get_outdir(args.case, args.n_experts, n_categories=1)


    demo_states = []
    demo_actions = []
    for i in range(args.n_experts):
        usr = User(time_steps = 128)
        states, actions = usr.generate_trajectory()
        demo_states.append(states)
        demo_actions.append(actions)

    #PathOfDemonstrationNpzFile = os.getcwd() + '/expert_trajectories.npz'

    np.savez(PathOfDemonstrationNpzFile, states=np.array(demo_states, dtype=object),
             actions=np.array(demo_actions, dtype=object)) 
    # command = ['python', 'customer_behaviour/algorithms/train_gym.py', args.algo, '--gpu', '-1', '--env', env, '--arch', 'FFSoftmax', '--steps', str(args.n_steps), '--load_demo', 'expert_trajectories.npz', '--update-interval', '128', '--entropy-coef' ,'0.01', '--outdir', outdir]
    # s = subprocess.Popen(command)

    run_algo(args)


def get_outdir(case, n_experts, n_categories):

    if not os.path.exists('results'): os.makedirs('results')

    path = join('results', case)
    if not os.path.exists(path): os.makedirs(path)

    path = join(path, str(n_experts) + '_expert(s)')
    if not os.path.exists(path): os.makedirs(path)

    path = join(path, str(n_categories) + '_product(s)')
    if not os.path.exists(path): os.makedirs(path)

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = join(path, time)

    return path


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
