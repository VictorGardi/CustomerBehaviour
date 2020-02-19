import gym
import collections
import numpy as np
from gym import spaces
from customer_behaviour.tools import dgm as dgm


DETERMINISTIC_SAMPLE = False

N_DEMOS_PER_EXPERT = 20


def categorize_age(age):
    if age < 30: return 0
    elif 30 <= age < 40: return 0.2
    elif 40 <= age < 50: return 0.4
    elif 50 <= age < 60: return 0.6
    elif 60 <= age < 70: return 0.8
    elif 70 <= age: return 1


class Case1():
    def __init__(self, model):
        self.model = model

    def get_spaces(self, n_historical_events):
        low = [0, 0] + n_historical_events * [0]
        high = [1, 1] + n_historical_events * [1]
        observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

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

        initial_state = [self.model.sex, categorize_age(self.model.age), *temp]

        return initial_state

    def get_step(self, state, action):
        history = state[2:]
        new_history = [*history[1:], action]

        new_state = [self.model.sex, categorize_age(self.model.age), *new_history]

        return new_state


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


class Case3():
    def __init__(self, model):
        self.model = model

    def get_sample(self, n_demos_per_expert, n_historical_events, n_time_steps):
        sample = self.model.sample2(n_demos_per_expert, n_historical_events, n_time_steps, n_product_groups=1)
        return sample

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiDiscrete(n_historical_events * [50])  # Assume that there is always less than 50 days between two consecutive purchases

        action_space = spaces.Discrete(2)

        return observation_space, action_space

    def get_action(self, receipt):
        action = 1 if receipt[0] > 0 else 0  # We only consider the first item
        return action

    def get_initial_state(self, history):
        # decode days between purchases
        temp = history[0, :]  # We only consider the first item
        assert temp[-1] > 0

        initial_state = []
        i = 0
        for h in reversed(temp):
            if h > 0:
                initial_state.append(i)
                i = 1
            else:
                i += 1

        return initial_state

    def get_step(self, state, action):
        if action == 0:
            # no purchase
            new_state = state.copy()
            new_state[0] += 1          
        else:
            # purchase
            new_state = [0, *state[:-1]]
            new_state[1] += 1
        return new_state


def define_case(case):
    switcher = {
        1: Case1,
        2: Case2,
        3: Case3
    }
    return switcher.get(case)


###################################################################
########## The code below should not need to be modified ##########
###################################################################


class DiscreteBuyingEvents(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(DiscreteBuyingEvents, self).__init__()
        self.model = dgm.DGM()
        self.n_time_steps = 0
        self.state = None


    def initialize_environment(self, case, n_historical_events, episode_length, n_demos_per_expert, agent_seed=None):
        temp = define_case(case)
        self.case = temp(self.model)

        self.n_historical_events = n_historical_events
        self.episode_length = episode_length
        self.n_demos_per_expert = n_demos_per_expert
        self.agent_seed = agent_seed

        self.observation_space, self.action_space = self.case.get_spaces(n_historical_events)


    def generate_expert_trajectories(self, n_experts, n_time_steps, out_dir, seed=True):
        states = []
        actions = []

        for i_expert in range(n_experts):
            self.model.spawn_new_customer(i_expert) if seed else model.spawn_new_customer()

            sample = self.case.get_sample(self.n_demos_per_expert, self.n_historical_events, n_time_steps)

            for subsample in sample:
                temp_states = []
                temp_actions = []

                history = subsample[0]
                data = subsample[1]

                initial_state = self.case.get_initial_state(history)

                self.state = initial_state

                temp_states.append(np.array(initial_state))  # the function step(action) returns the state as an np.array

                for i, receipt in enumerate(data.T, start=1):  # transpose since the receipts are columns in data
                    action = self.case.get_action(receipt)
                    temp_actions.append(action)

                    if i == data.shape[1]:
                        # The number of states and actions must be equal
                        pass
                    else:
                        state, _, _, _ = self.step(action)
                        temp_states.append(state)

                states.append(temp_states)
                actions.append(temp_actions)

        np.savez(out_dir + '/expert_trajectories.npz', states=np.array(states, dtype=object),
             actions=np.array(actions, dtype=object))

        return {'states': states, 'actions': actions}


    def seed(self, seed=None):
        pass


    def step(self, action):        
        new_state = self.case.get_step(self.state, action)
        self.state = new_state
        self.n_time_steps += 1
        done = self.n_time_steps >= self.episode_length
        reward = 0
        return np.array(self.state), reward, done, {}


    def reset(self):
        # Reset the state of the environment to an initial state
        self.model.spawn_new_customer(self.agent_seed)

        sample = self.case.get_sample(self.n_demos_per_expert, self.n_historical_events, 0)
        # sample is an array of tuples (history, data) of length n_demos_per_expert
        # choose a random history
        i = np.random.randint(0, self.n_demos_per_expert)
        history, _ = sample[i]
        
        self.state = self.case.get_initial_state(history)
        self.n_time_steps = 0
        return np.array(self.state)
    

    def render(self, mode='human', close=False):
        pass


###########################
########## Trash ##########
###########################


'''
sample = self.model.sample(n_demos_per_expert * (self.n_historical_events + n_time_steps))

for subsample in np.split(sample, n_demos_per_expert, axis=1):
    temp_states = []
    temp_actions = []

    history = subsample[:, :self.n_historical_events]
    initial_state = self.case.get_initial_state(history)

    self.state = initial_state

    temp_states.append(np.array(initial_state))  # the function step(action) returns the state as an np.array

    i = self.n_historical_events
    while i < subsample.shape[1]:
        receipt = subsample[:, i]
        action = self.case.get_action(receipt)
        temp_actions.append(action)

        if i == subsample.shape[1] - 1:
            # The number of states and actions must be equal
            pass
        else:
            state, _, _, _ = self.step(action)
            temp_states.append(state)

        i += 1

    states.append(temp_states)
    actions.append(temp_actions)
'''

'''
class SpaceSwitcher():
    def __init__(self, n_historical_events):
        self.n_historical_events = n_historical_events

    def case_to_space(self, case):
        method_name = '_case_' + str(case)
        method = getattr(self, method_name)
        return method()

    def _case_1(self):
        low = [0, 0] + self.n_historical_events * [0]
        high = [1, 1] + self.n_historical_events * [1]
        observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        action_space = spaces.Discrete(2)

        return observation_space, action_space


class ActionSwitcher():
    def __init__(self):
        pass

    def case_to_action(self, case, receipt):
        method_name = '_case_' + str(case)
        method = getattr(self, method_name)
        return method(receipt)

    def _case_1(self, receipt):
        action = 1 if receipt[0] > 0 else 0  # We only consider the first item
        return action


class InitStateSwitcher():
    def __init__(self, model, history):
        self.model = model
        self.history = history

    def case_to_init_state(self, case):
        method_name = '_case_' + str(case)
        method = getattr(self, method_name)
        return method()

    def _case_1(self):
        history = self.history[0, :]  # We only consider the first item
        history[history > 0] = 1

        initial_state = [self.model.sex, categorize_age(self.model.age), *history]

        return initial_state


class StepSwitcher():
    def __init__(self, model):
        self.model = model

    def case_to_step(self, case, state, action):
        method_name = '_case_' + str(case)
        method = getattr(self, method_name)
        return method(state, action)

    def _case_1(self, state, action):
        history = state[2:]
        new_history = [*history[1:], action]

        new_state = [self.model.sex, categorize_age(self.model.age), *new_history]

        return new_state
'''