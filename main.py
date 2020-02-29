"""An example of training PPO against OpenAI Gym Envs.

This script is an example of training a PPO agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_ppo_gym.py --env CartPole-v0
"""
import os
import argparse
from customer_behaviour.tools.tools import get_env, get_outdir, str2bool, move_dir
import numpy as np
import gym
import custom_gym
import gym.wrappers

import chainer
from chainer import functions as F
import chainerrl
# from customer_behaviour import chainerrl

from chainerrl.agents import a3c
from chainerrl.agents import PPO
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl import policies


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64,64)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(64,64)):
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

def convert_logits_to_probs(logits):
    odds = np.exp(logits)
    probs = odds/(np.sum(odds))
    return probs


def save_agent_demo(env, agent, out_dir, max_t=10000):
    import numpy as np
    r, t = 0, 0
    agent_observations = []
    agent_actions = []
    action_probs = []
    while t < max_t:
        agent_observations.append([])
        agent_actions.append([])
        action_probs.append([])
        obs = env.reset()
        while True:
            b_state = agent.batch_states([obs], agent.xp, agent.phi)
            logits = agent.model(b_state)[0].logits._data[0]
            probs = convert_logits_to_probs(logits[0,:])
            act = agent.act(obs)
            agent_observations[-1].append(obs)
            agent_actions[-1].append(act)
            action_probs[-1].append(probs)
            obs, reward, done, _ = env.step(act)
            t += 1
            r += reward
            if done or t >= max_t:
                print(t)
                break

    # save numpy array consists of lists
    np.savez(out_dir+'/trajectories.npz', states=np.array(agent_observations, dtype=object),
             actions=np.array(agent_actions, dtype=object))
    np.savez(out_dir+'/action_probs.npz', action_probs = np.array(action_probs, dtype=object))


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('algo', default='gail', choices=['gail', 'airl'], type=str)
    parser.add_argument('--case', type=str, default='discrete_events')
    parser.add_argument('--n_experts', type=int, default=1)
    parser.add_argument('--n_demos_per_expert', type=int, default=10)
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
    parser.add_argument('--bound-mean', action='store_true', default=False)  # only for FFGaussian
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    #parser.add_argument('--outdir', type=str, default='results', help='Directory path to save output files.'' If it does not exist, it will be created.')
    
    parser.add_argument('--eval_episode_length', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=10000)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)  # does not make sense since we do not have any reward signal
    parser.add_argument('--standardize-advantages', action='store_true', default=True)  # True is default in PPO
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    #parser.add_argument('--load_demo', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
    parser.add_argument('--monitor', action='store_true', default=False)
    parser.add_argument('--update-interval', type=int, default=1024)
    parser.add_argument('--batchsize', type=int, default=64)  # mini-batch size
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
        if test:
            episode_length = args.eval_episode_length
        else:
            episode_length = args.episode_length

        env.initialize_environment(args.state_rep, args.n_historical_events, episode_length, 1, args.length_expert_TS, args.agent_seed)

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
    sample_env.initialize_environment(args.state_rep, args.n_historical_events, args.episode_length, 1, args.length_expert_TS, args.agent_seed)
    demonstrations = sample_env.generate_expert_trajectories(args.n_experts, out_dir=dst, seed=args.seed_expert, eval=False)
    timestep_limit = sample_env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')  # This value is None


    # Generate expert data for evaluation
    temp_env = gym.make(args.env)
    temp_env.initialize_environment(
        args.state_rep, 
        args.n_historical_events, 
        args.episode_length,  # This parameter does not really matter since we create this env only for generating samples 
        args.n_demos_per_expert,  # How large should the expert cluster be?
        args.eval_episode_length, 
        args.agent_seed
    )
    temp_env.generate_expert_trajectories(args.n_experts, out_dir=dst, seed=args.seed_expert, eval=True)

    
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    # Normalize observations based on their empirical mean and variance
    # obs_normalizer = None
    if args.state_rep == 1:
        shape = obs_space.low.size
    elif args.state_rep == 2:
        shape = obs_space.n
    elif args.state_rep == 3:
        shape = obs_space.nvec.size
    else:
        raise NotImplementedError
    obs_normalizer = chainerrl.links.EmpiricalNormalization(shape, clip_threshold=5)
    #chainerrl.links.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)

    # Switch policy types accordingly to action space types
    if args.arch == 'FFSoftmax':
        if args.state_rep == 1:
            model = A3CFFSoftmax(obs_space.low.size, action_space.n, hidden_sizes=(args.batchsize, args.batchsize))      # Should it really be batchsize here???
        elif args.state_rep == 2:
            model = A3CFFSoftmax(obs_space.n, action_space.n, hidden_sizes=(args.batchsize, args.batchsize))
        elif args.state_rep == 3:
            model = A3CFFSoftmax(obs_space.nvec.size, action_space.n)
        else:
            raise NotImplementedError
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
        demonstrations = np.load(dst + '/expert_trajectories.npz')
        D = Discriminator(gpu=args.gpu, n_units=args.batchsize)
        agent = Agent(demonstrations=demonstrations, discriminator=D,
                      model=model, optimizer=opt,
                      obs_normalizer=obs_normalizer,
                      gpu=args.gpu,
                      update_interval=args.update_interval,
                      minibatch_size=args.batchsize, epochs=args.epochs,
                      clip_eps_vf=None, entropy_coef=args.entropy_coef,
                      standardize_advantages=args.standardize_advantages,)

    if args.load:
        # By default, not in here
        agent.load(args.load)

    if args.demo:
        # By default, not in here
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
            env=make_env(False),                    # Environment train the agent against (False -> scaled rewards)
            eval_env=make_env(True),                # Environment used for evaluation
            outdir=args.outdir,
            steps=args.steps,                       # Total number of timesteps for training (args.n_training_episodes*args.episode_length)
            eval_n_steps=None,                      # Number of timesteps at each evaluation phase
            eval_n_episodes=args.eval_n_runs,       # Number of episodes at each evaluation phase (default: 10)
            eval_interval=args.eval_interval,       # Interval of evaluation (defualt: 10000 steps (?))
            train_max_episode_len=timestep_limit,   # Maximum episode length during training (is None)
            save_best_so_far_agent=False,
            step_hooks=[
                lr_decay_hook,
                clip_eps_decay_hook,
            ],
        )
        save_agent_demo(make_env(True), agent, args.outdir, 10 * args.eval_episode_length)  # originally it was make_env(test=False) which seems strange
    
    # Move result files to correct folder and remove empty folder
    move_dir(args.outdir, dst)
    os.rmdir(args.outdir)


if __name__ == '__main__':
    main()

