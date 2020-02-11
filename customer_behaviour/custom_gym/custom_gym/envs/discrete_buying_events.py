import gym
import collections
import numpy as np
from gym import spaces
from customer_behaviour.tools import dgm as dgm


def categorize_age(age):
    if age < 30: return 0
    elif 30 <= age < 40: return 0.2
    elif 40 <= age < 50: return 0.4
    elif 50 <= age < 60: return 0.6
    elif 60 <= age < 70: return 0.8
    elif 70 <= age: return 1


class DiscreteBuyingEvents(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DiscreteBuyingEvents, self).__init__()
        
        self.model = dgm.DGM()

        self.n_time_steps = 0

        self.state = None


    def initialize_environment(self, n_products, n_historical_events, episode_length, agent_seed=None):
        # The implementaiton of this function depends on the chosen state representation

        self.episode_length = episode_length

        self.n_products = n_products
        self.n_historical_events = n_historical_events

        self.agent_seed = agent_seed

        if self.n_products == 1:
            # There are only two discrete actions: "buy something" or "do not buy something"
            self.action_space = spaces.Discrete(2)

            # The state consists of a customer's sex, age and purchase history
            low = [0, 0] + self.n_historical_events * [0]
            high = [1, 1] + self.n_historical_events * [1]
            self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        else:
            # The action corresponds to a full receipt
            raise NotImplementedError


    def generate_expert_trajectories(self, n_experts, n_time_steps, out_dir, seed=True):
        states = []
        actions = []

        for i_expert in range(n_experts):

            temp_states = []
            temp_actions = []

            self.model.spawn_new_customer(i_expert) if seed else model.spawn_new_customer()
            # sample = self.model.sample(self.n_historical_events + n_time_steps)

            sample = np.zeros((6, self.n_historical_events + n_time_steps))

            ones = [1, 1, 1, 1, 1, 1]

            for i in np.arange(0.1, 1.0, 0.1):
                j = int(i * (self.n_historical_events + n_time_steps))
                sample[:, j] = ones

            history = sample[:, :self.n_historical_events]        
            initial_state = self.initialize_state(history)
            
            self.state = initial_state

            temp_states.append(np.array(initial_state))  # the function step(action) returns the state as an np.array

            i = self.n_historical_events
            while i < sample.shape[1]:
                if isinstance(self.action_space, spaces.Discrete):
                    # There are only two discrete actions: "buy something" or "do not buy something"
                    action = 1 if sample[0, i] > 0 else 0  # We only consider one item
                elif isinstance(self.action_space, spaces.Box):
                    raise NotImplementedError

                temp_actions.append(action)

                if i == sample.shape[1] - 1:
                    # The number of states and actions must be equal
                    pass
                else:
                    state, _, _, _ = self.step(action)
                    temp_states.append(state)

                i += 1

            states.append(temp_states)
            actions.append(temp_actions)

        np.savez(out_dir + '/expert_trajectories.npz', states=np.array(states, dtype=object),
             actions=np.array(actions, dtype=object))

        return {'states': states, 'actions': actions}


    def initialize_state(self, history):
        # The implementaiton of this function depends on the chosen state representation

        if self.n_products == 1:
            history = np.sum(history, axis=0)
            history[history > 0] = 1

            initial_state = [self.model.sex, categorize_age(self.model.age), *history]
        else:
            raise NotImplementedError

        return initial_state


    def seed(self, seed=None):
        pass


    def step(self, action):
        # The implementaiton of this function depends on the chosen state representation

        if self.n_products == 1:
            history = self.state[2:]
            new_history = [*history[1:], action]

            new_state = [self.model.sex, categorize_age(self.model.age), *new_history]

            self.state = new_state

            self.n_time_steps += 1

            done = self.n_time_steps >= self.episode_length

            reward = 0
        else:
            raise NotImplementedError

        return np.array(self.state), reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state

        self.model.spawn_new_customer(self.agent_seed)

        sample = self.model.sample(self.n_historical_events)

        self.state = self.initialize_state(sample)

        self.n_time_steps = 0

        return np.array(self.state)
    

    def render(self, mode='human', close=False):
        pass
    
