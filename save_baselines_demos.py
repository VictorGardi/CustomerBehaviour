import gym
import os
import datetime
import custom_gym
import numpy as np  # 1.16.4
import tensorflow as tf  # 1.14.0
import matplotlib.pyplot as plt
from stable_baselines import GAIL
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():
    env = gym.make('discrete-buying-events-v0')
    env.initialize_environment(case = 2, n_historical_events = 96, episode_length = 256, n_demos_per_expert=1, n_expert_time_steps=256, agent_seed=0)
    env = DummyVecEnv([lambda: env])
    agent = GAIL.load(os.getcwd() + '/stable-baselines-test/gail')
    episode_length = 1024
    save_agent_demo(env, agent, 'stable-baselines-test', 10*episode_length)


def save_agent_demo(env, agent, out_dir, max_t=10000):
    t = 0
    agent_observations = []
    agent_actions = []
    while t < max_t:
        agent_observations.append([])
        agent_actions.append([])
        obs = env.reset()

        while True:
            act, _ = agent.predict(obs)
            agent_observations[-1].append(obs[0])
            agent_actions[-1].append(act[0])
            obs, _, done, _ = env.step(act)
            t += 1
            if done or t >= max_t:
                print(t)
                break
        
    
    np.savez(out_dir+'/trajectories.npz', states=np.array(agent_observations, dtype=object),
             actions=np.array(agent_actions, dtype=object))


if __name__ == '__main__':
	main()
