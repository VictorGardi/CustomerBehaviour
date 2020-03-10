import numpy as np
import gym
import itertools
import custom_gym
import matplotlib.pyplot as plt
from customer_behaviour.tools import dgm as dgm
import os
import gym
import custom_gym

from stable_baselines import SAC
from stable_baselines.gail import ExpertDataset, GAIL


def main():
    path = os.getcwd() + '/stable-baselines-test/eval_expert_trajectories.npz'
    save_experts(n_experts=1, episode_length = 256, n_historical_events = 96, path = path)
    env_name = 'discrete-buying-events-v0'
    # Load the expert dataset
    dataset = ExpertDataset(expert_path=path, traj_limitation=10, verbose=1)
    env = gym.make(env_name)
    env.initialize_environment(case = 21, n_historical_events = 96, episode_length = 512, n_demos_per_expert=1, n_expert_time_steps=256, agent_seed=0)
    model = GAIL('MlpPolicy', env, dataset, verbose=1, tensorboard_log="./test_tensorboard/")
    # Note: in practice, you need to train for 1M steps to have a working policy
    model.learn(total_timesteps=100)
    model.save("gail")

    del model # remove to demonstrate saving and loading

    #model = GAIL.load("gail_sb")
    #print(model)

    #env = gym.make(env_name)
    #print('Initializing env...')
    #env.initialize_environment(case = 2, n_historical_events = 96, episode_length = 256, n_demos_per_expert=1, n_expert_time_steps=256, agent_seed=0)
    #obs = env.reset()
    #print('Starting while loop...')
    #while True:
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
    #    env.render()

    
############################
##### Helper functions #####
############################


def save_experts(n_experts, seed = 0, episode_length = 256, n_historical_events = 96, path = None):
    model = dgm.DGM()
    case = Case2(model=model)

    states, actions = get_states_actions(case, model, seed, episode_length, n_historical_events)
    states = np.array(states[0], dtype=float)
    actions = np.array(actions[0], dtype=float)
    rewards = np.zeros_like(actions)
    episode_starts = np.zeros_like(actions)
    episode_starts[0] = 1
    n_episodes = int(np.sum(episode_starts))
    print(n_episodes)
    episode_returns = np.array(n_episodes*[0])
    

    if n_experts == 1:
        np.savez(path, 
                obs=states,
                actions=actions,
                rewards=rewards,
                episode_starts=episode_starts,
                episode_returns=episode_returns)
    else:
        mat_states = states
        mat_actions = actions
        for expert in range(n_experts):
            states, actions = get_states_actions(case, model, seed, episode_length, n_historical_events)

            states = np.array(states[0], dtype=object)
            actions = np.array(actions[0], dtype=object)
            mat_states = np.dstack((mat_states, states))
            mat_actions = np.dstack((mat_actions, actions))

        np.savez(path, 
                obs=mat_states,
                actions=mat_actions,
                rewards=rewards,
                episode_starts=episode_starts,
                episode_returns=episode_returns)

        

def get_states_actions(case, model, seed, episode_length = 256, n_historical_events = 96, save_for_visualisation=False):
    states = []
    actions = []

    model.spawn_new_customer(seed)
    sample = case.get_sample(1, n_historical_events, episode_length)

    for subsample in sample:
        temp_states = []
        temp_actions = []

        history = subsample[0]
        data = subsample[1]

        initial_state = case.get_initial_state(history)

        state = initial_state

        temp_states.append(np.array(initial_state))  # the function step(action) returns the state as an np.array

        for i, receipt in enumerate(data.T, start=1):  # transpose since the receipts are columns in data
            action = case.get_action(receipt)
            temp_actions.append(action)

            if i == data.shape[1]:
                # The number of states and actions must be equal
                pass
            else:
                state = step(case, state, action)
                temp_states.append(state)

        states.append(temp_states)
        actions.append(temp_actions)
        if save_for_visualisation:
            np.savez('stable-baselines-test/eval_expert_trajectories.npz', states=states, actions=actions)

    return states, actions

def step(case, state, action):        
    new_state = case.get_step(state, action)
    state = new_state
    return np.array(state)


#################
##### Cases #####
#################

class Case2():
    def __init__(self, model):
        self.model = model

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(n_historical_events)

        action_space = spaces.Discrete(2)

        return observation_space, action_space

    def get_sample(self, n_demos_per_expert, n_historical_events, n_time_steps):
        temp_sample = self.model.sample(n_demos_per_expert * (n_historical_events + n_time_steps))
        sample = []
        for subsample in np.split(temp_sample, n_demos_per_expert, axis=1):
            history = subsample[:, :n_historical_events]
            data = subsample[:, n_historical_events:]
            sample.append((history, data))
        return sample

    def get_action(self, receipt):
        action = 1 if receipt[0] > 0 else 0  # We only consider the first item
        return action

    def get_initial_state(self, history):
        temp = history[0, :].copy()  # We only consider the first item

        temp[temp > 0] = 1

        initial_state = temp

        return initial_state

    def get_step(self, state, action):
        new_state = [*state[1:], action]
        return new_state

class Case21():
    def __init__(self, model):
        self.model = model

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(n_historical_events)

        action_space = spaces.Discrete(2)

        return observation_space, action_space

    def get_sample(self, n_demos_per_expert, n_historical_events, n_time_steps):
        temp_sample = self.model.sample(n_demos_per_expert * (n_historical_events + n_time_steps))
        sample = []
        for subsample in np.split(temp_sample, n_demos_per_expert, axis=1):
            history = subsample[:, :n_historical_events]
            data = subsample[:, n_historical_events:]
            sample.append((history, data))
        return sample

    def get_action(self, receipt):
        action = 1 if np.any(np.nonzero(receipt)) else 0
        return action

    def get_initial_state(self, history):
        temp = history.copy()
        temp = np.sum(temp, axis=0)

        temp[temp > 0] = 1

        initial_state = temp

        return initial_state

    def get_step(self, state, action):
        new_state = [*state[1:], action]
        return new_state

if __name__ == '__main__':
    main()
