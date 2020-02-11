"""An example of training PPO against OpenAI Gym Envs.

This script is an example of training a PPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_ppo_gym.py --env CartPole-v0
"""
import os
import argparse
from customer_behaviour.tools.tools import get_env, get_outdir, str2bool, move_dir

import gym
import custom_gym
import gym.wrappers

import chainer
from chainer import functions as F
import chainerrl

from chainerrl.agents import a3c
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFGaussian(chainer.Chain, a3c.A3CModel):
    """An example of A3C feedforward Gaussian policy."""

    def __init__(self, obs_size, action_space,
                 n_hidden_layers=2, n_hidden_channels=64,
                 bound_mean=None):
        assert bound_mean in [False, True]
        super().__init__()
        hidden_sizes = (n_hidden_channels,) * n_hidden_layers
        with self.init_scope():
            self.pi = policies.FCGaussianPolicyWithStateIndependentCovariance(
                obs_size, action_space.low.size,
                n_hidden_layers, n_hidden_channels,
                var_type='diagonal', nonlinearity=F.tanh,
                bound_mean=bound_mean,
                min_action=action_space.low, max_action=action_space.high,
                mean_wscale=1e-2)
            self.v = links.MLP(obs_size, 1, hidden_sizes=hidden_sizes)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


def save_agent_demo(env, agent, out_dir, max_t=2000):
    import numpy as np
    r, t = 0, 0
    agent_observations = []
    agent_actions = []
    while t < max_t:
        agent_observations.append([])
        agent_actions.append([])
        obs = env.reset()
        while True:
            act = agent.act(obs)
            agent_observations[-1].append(obs)
            agent_actions[-1].append(act)
            obs, reward, done, _ = env.step(act)
            t += 1
            r += reward
            if done or t >= max_t:
                print(t)
                break
                print('exiting...')
                exit()

    # save numpy array consists of lists
    np.savez(out_dir+'/trajectories.npz', states=np.array(agent_observations, dtype=object),
             actions=np.array(agent_actions, dtype=object))


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', default='gail', choices=['gail', 'airl'], type=str)
    parser.add_argument('--case', type=str, default='discrete_events')
    parser.add_argument('--n_experts', type=int, default=1)
    parser.add_argument('--state_rep', type=int, default=1)
    parser.add_argument('--length_expert_TS', type=int, default=100)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--n_training_episodes', type=int, default=1000)
    parser.add_argument('--seed_expert', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate expert seed mode.")
    parser.add_argument('--agent_seed', type=int, default=None)
    
    parser.add_argument('--n_historical_events', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--arch', type=str, default='FFSoftmax',
                        choices=('FFSoftmax', 'FFMellowmax',
                                 'FFGaussian'))
    parser.add_argument('--bound-mean', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    #parser.add_argument('--outdir', type=str, default='results', help='Directory path to save output files.'' If it does not exist, it will be created.')
    
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--standardize-advantages', action='store_true')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    #parser.add_argument('--load_demo', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--update-interval', type=int, default=128)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    args = parser.parse_args()
    args.outdir = get_outdir(args.algo, args.case, args.n_experts, args.state_rep)
    args.env = get_env(args.case, args.n_experts)  
    args.steps = args.n_training_episodes*args.episode_length

    logging.basicConfig(level=args.logger_level)

    # Set a random seed used in ChainerRL
    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    if not (args.demo and args.load):
        args.outdir = experiments.prepare_output_dir(args, args.outdir)
    temp = args.outdir.split('/')[-1]
    dst = args.outdir.strip(temp)

    def make_env(test):
        env = gym.make(args.env)
        env.initialize_environment(args.state_rep, args.n_historical_events, args.episode_length, args.agent_seed)

        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if not test:
            # Scale rewards (and thus returns) to a reasonable range so that
            # training is easier
            env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    sample_env = gym.make(args.env)
    sample_env.initialize_environment(args.state_rep, args.n_historical_events, args.episode_length, args.agent_seed)
    demonstrations = sample_env.generate_expert_trajectories(args.n_experts, args.length_expert_TS, out_dir=dst, seed=args.seed_expert)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    ####-----create_expert data here and take args.seed_expert as input and save as expert_trajectories.npz 
    # file in dst that file path is our demonstrations variable ----

    # Normalize observations based on their empirical mean and variance

    obs_normalizer = None                                                                # HarDKODAT
    #chainerrl.links.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)

    # Switch policy types accordingly to action space types
    if args.arch == 'FFSoftmax':
        model = A3CFFSoftmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A3CFFMellowmax(obs_space.low.size, action_space.n)
    elif args.arch == 'FFGaussian':
        model = A3CFFGaussian(obs_space.low.size, action_space,
                              bound_mean=args.bound_mean)

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))
    if args.algo == 'ppo':
        agent = PPO(model, opt,
                    obs_normalizer=obs_normalizer,
                    gpu=args.gpu,
                    update_interval=args.update_interval,
                    minibatch_size=args.batchsize, epochs=args.epochs,
                    clip_eps_vf=None, entropy_coef=args.entropy_coef,
                    standardize_advantages=args.standardize_advantages,
                    )
    elif args.algo == 'gail':
        import numpy as np
        from customer_behaviour.algorithms.irl.gail import GAIL
        from customer_behaviour.algorithms.irl.gail import Discriminator
        #demonstrations = np.load(args.load_demo)
        D = Discriminator(gpu=args.gpu)
        agent = GAIL(demonstrations=demonstrations, discriminator=D,
                     model=model, optimizer=opt,
                     obs_normalizer=obs_normalizer,
                     gpu=args.gpu,
                     update_interval=args.update_interval,
                     minibatch_size=args.batchsize, epochs=args.epochs,
                     clip_eps_vf=None, entropy_coef=args.entropy_coef,
                     standardize_advantages=args.standardize_advantages,)
    elif args.algo == 'airl':
        import numpy as np
        from customer_behaviour.algorithms.irl.airl import AIRL as Agent
        from customer_behaviour.algorithms.irl.airl import Discriminator
        # obs_normalizer = None
        #demonstrations = np.load(args.load_demo)
        D = Discriminator(gpu=args.gpu)
        agent = Agent(demonstrations=demonstrations, discriminator=D,
                      model=model, optimizer=opt,
                      obs_normalizer=obs_normalizer,
                      gpu=args.gpu,
                      update_interval=args.update_interval,
                      minibatch_size=args.batchsize, epochs=args.epochs,
                      clip_eps_vf=None, entropy_coef=args.entropy_coef,
                      standardize_advantages=args.standardize_advantages,)

    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
        outdir = args.load if args.load else args.outdir
        save_agent_demo(make_env(False), agent, outdir)
    else:
        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        lr_decay_hook = experiments.LinearInterpolationHook(
            args.steps, args.lr, 0, lr_setter)

        # Linearly decay the clipping parameter to zero
        def clip_eps_setter(env, agent, value):
            agent.clip_eps = max(value, 1e-8)

        clip_eps_decay_hook = experiments.LinearInterpolationHook(
            args.steps, 0.2, 0, clip_eps_setter)

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=make_env(False),
            eval_env=make_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            train_max_episode_len=timestep_limit,
            save_best_so_far_agent=False,
            step_hooks=[
                lr_decay_hook,
                clip_eps_decay_hook,
            ],
        )
        save_agent_demo(make_env(False), agent, args.outdir)
    
    # Move result files to correct folder and remove empty folder
    move_dir(args.outdir, dst)
    os.rmdir(args.outdir)

    # Visualization


if __name__ == '__main__':
    main()

