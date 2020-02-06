import numpy as np
import gym
import custom_gym
from stable_baselines.common.env_checker import check_env

env = gym.make('discrete-buying-events-v0', age=0, sex=0, history=20*[0])
check_env(env)




#data = np.load('trajectories.npz', allow_pickle=True)
#for item in data.files:
#    print(item)
#    print(data[item])