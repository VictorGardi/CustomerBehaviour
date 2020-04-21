import gym
import random
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

def categorize_age2(age):
    if age < 30: return 0
    elif 30 <= age < 40: return 1
    elif 40 <= age < 50: return 2
    elif 50 <= age < 60: return 3
    elif 60 <= age < 70: return 4
    elif 70 <= age: return 5

def categorize_amount(amount):
    if amount == 0: 
        return 0
    elif 0 < amount < 100: 
        return 1
    elif 100 <= amount < 200: 
        return 2
    elif 200 <= amount < 300: 
        return 3
    elif 300 <= amount < 400: 
        return 4
    elif 400 <= amount < 500: 
        return 5
    elif 500 <= amount < 600: 
        return 6
    elif 600 <= amount < 700: 
        return 7
    elif 700 <= amount < 800: 
        return 8
    elif 800 <= amount < 900: 
        return 9
    else:
        # Amount > 900
        return 10

class Case1():
    def __init__(self, model, n_experts=None, adam_days=None):
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

    def get_initial_state(self, history, seed=None):
        temp = history[0, :].copy()  # We only consider the first item
        temp[temp > 0] = 1

        initial_state = [self.model.sex, categorize_age(self.model.age), *temp]

        return initial_state

    def get_step(self, state, action):
        history = state[2:]
        new_history = [*history[1:], action]

        new_state = [self.model.sex, categorize_age(self.model.age), *new_history]

        return new_state

class Case11():
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model

    def get_spaces(self, n_historical_events):
        sex_age = [2, 6]
        history = n_historical_events * [1]
        space = sex_age + history
        observation_space = spaces.MultiDiscrete(space)

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed=None):
        temp = history.copy()
        temp = np.sum(temp, axis=0)

        temp[temp > 0] = 1

        initial_state = [self.model.sex, categorize_age2(self.model.age), *temp]

        return initial_state

    def get_step(self, state, action):
        history = state[2:]
        new_history = [*history[1:], action]

        new_state = [self.model.sex, categorize_age2(self.model.age), *new_history]

        return new_state


class Case2():
    def __init__(self, model, n_experts=None, adam_days=None):
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

    def get_initial_state(self, history, seed=None):
        temp = history[0, :].copy()  # We only consider the first item

        temp[temp > 0] = 1

        initial_state = temp

        return initial_state

    def get_step(self, state, action):
        new_state = [*state[1:], action]
        return new_state


class Case21():
    def __init__(self, model, n_experts=None, adam_days=None):
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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed=None):
        temp = history.copy()
        temp = np.sum(temp, axis=0)

        temp[temp > 0] = 1

        initial_state = temp

        return initial_state

    def get_step(self, state, action):
        new_state = [*state[1:], action]
        return new_state


class Case22():  # dummy encoding (dynamic)
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.n_experts + n_historical_events) 

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = state[self.n_experts:]
        new_state = [*dummy, *history[1:], action]
        return new_state


class Case221():  # dummy encoding (dynamic) for generator but no dummy for discrinator
    # + let discriminator compare expert and discriminator from same class
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.n_experts + n_historical_events) 

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = state[self.n_experts:]
        new_state = [*dummy, *history[1:], action]
        return new_state


class Case222(): # dummy encoding (dynamic) for generator but no dummy for discrinator
    # + let discriminator compare expert and discriminator from same class but not from one expert at a time
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.n_experts + n_historical_events) 

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = state[self.n_experts:]
        new_state = [*dummy, *history[1:], action]
        return new_state
     
class Case23():  # Consider purchase amounts
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiDiscrete(self.n_experts * [2] + n_historical_events * [11])

        action_space = spaces.Discrete(11)

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
        action = categorize_amount(np.sum(receipt))
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp = [categorize_amount(x) for x in temp]

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = state[self.n_experts:]
        new_state = [*dummy, *history[1:], action]
        return new_state


class Case24():  # dummy encoding (fixed)
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(10 + n_historical_events) 

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        dummy = np.zeros(10)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:10]
        history = state[10:]
        new_state = [*dummy, *history[1:], action]
        return new_state


class Case3():  # ÄR DET ETT PROBLEM ATT VI SÄTTER 50 SOM MAX? MINNS RESULTAT ENDAST [1, 1, ..., 1, 1]
    def __init__(self, model, n_experts=None, adam_days=None):
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

    def get_initial_state(self, history, seed=None):
        # Extract number of elapsed days between purchases
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

class Case31():
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiDiscrete(10 * [1] + n_historical_events * [10000])

        action_space = spaces.Discrete(2)

        return observation_space, action_space

    def get_sample(self, n_demos_per_expert, n_historical_events, n_time_steps):
        sample = self.model.sample3(n_demos_per_expert, n_historical_events, n_time_steps)
        return sample

    def get_action(self, receipt):
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, seed=None):
        # Extract number of elapsed days between purchases

        temp = np.sum(history, axis=0)
        temp[temp > 0] = 1

        assert temp[-1] > 0

        initial_state = []
        i = 0
        for h in reversed(temp):
            if h > 0:
                initial_state.append(i)
                i = 1
            else:
                i += 1
        initial_state[0] = 1

        dummy = np.zeros(10)
        dummy[seed] = 1

        initial_state = np.concatenate((initial_state, dummy))

        return initial_state

    def get_step(self, state, action):
        dummy = state[-10:]
        history = state[:-10].copy()

        if action == 0:
            # no purchase
            new_history = history
            new_history[0] += 1          
        else:
            # purchase
            new_history = [1, *history[:-1]]
            # new_history[1] += 1

        new_state = [*new_history, *dummy]

        return new_state


class Case7():
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts
        self.adam_days = adam_days

        self.adam_baskets = []

        self.N = 10  # items in basket

        for i in range(n_experts):
            self.model.spawn_new_customer(i)
            sample = self.model.sample3(self.N, self.adam_days + 1, 0)

            temp_basket = []

            for s in sample:
                temp = np.sum(s[0], axis=0)
                temp[temp > 0] = 1
                assert temp[-1] > 0

                adam = []
                j = 0
                for t in reversed(temp):
                    if t > 0:
                        adam.append(j)
                        j = 1
                    else:
                        j += 1
                adam[0] = 1

                temp_basket.append(adam[1:])

            self.adam_baskets.append(temp_basket)

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.adam_days + n_historical_events) 

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
        action = 1 if np.count_nonzero(receipt) > 0 else 0
        return action

    def get_initial_state(self, history, adam):
        temp = np.sum(history, axis=0)

        temp[temp > 0] = 1

        initial_state = np.concatenate((adam, temp))

        return initial_state

    def get_step(self, state, action):  # ska man byta till annan adam här??? initerar agent med random adam så måste måste finnas bland expertdemos?
        adam = state[:self.adam_days]
        history = state[self.adam_days:]
        new_state = [*adam, *history[1:], action]
        return new_state


class Case4():  # [dummy + product 1 + product 2]
    def __init__(self, model, n_experts=None, adam_days=None):
        self.model = model
        self.n_experts = n_experts

    def get_spaces(self, n_historical_events):
        observation_space = spaces.MultiBinary(self.n_experts + 2 * n_historical_events) 

        action_space = spaces.Discrete(4)

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
        purchase_product1 = np.count_nonzero(receipt[:3]) > 0
        purchase_product2 = np.count_nonzero(receipt[3:]) > 0

        if purchase_product1 and purchase_product2:
            action = 0
        elif purchase_product1 and not purchase_product2:
            action = 1
        elif not purchase_product1 and purchase_product2:
            action = 2
        else:
            action = 3

        return action

    def get_initial_state(self, history, seed):
        temp1 = np.sum(history[:3, :], axis=0)
        temp2 = np.sum(history[3:, :], axis=0)

        temp1[temp1 > 0] = 1
        temp2[temp2 > 0] = 1

        dummy = np.zeros(self.n_experts)
        dummy[seed] = 1

        initial_state = np.concatenate((dummy, temp1, temp2))

        return initial_state

    def get_step(self, state, action):
        dummy = state[:self.n_experts]
        history = np.split(np.array(state[self.n_experts:]), 2)
        history1 = history[0]
        history2 = history[1]

        if action == 0:
            a1 = 1
            a2 = 1
        elif action == 1:
            a1 = 1
            a2 = 0
        elif action == 2:
            a1 = 0
            a2 = 1
        else:
            a1 = 0
            a2 = 0

        new_state = [*dummy, *history1[1:], a1, *history2[1:], a2]
        return new_state

def define_case(case):
    switcher = {
        1: Case1,
        11: Case11,
        2: Case2,
        21: Case21,
        22: Case22,
        221: Case221,
        222: Case222,
        23: Case23,
        24: Case24,
        3: Case3,
        31: Case31,
        4: Case4,
        7: Case7
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


    def initialize_environment(
            self, 
            case, 
            n_historical_events, 
            episode_length, 
            n_experts, 
            n_demos_per_expert, 
            n_expert_time_steps, 
            seed_agent=True, 
            seed_expert=True,
            rank=None,
            n_processes=None,
            adam_days=None
            ):
        temp = define_case(case)
        self.case = temp(self.model, n_experts, adam_days)

        self.n_historical_events = n_historical_events
        self.episode_length = episode_length
        self.n_experts = n_experts
        self.n_expert_time_steps = n_expert_time_steps
        self.n_demos_per_expert = n_demos_per_expert
        self.seed_agent = seed_agent
        self.seed_expert = seed_expert

        self.rank = rank
        self.n_processes = n_processes

        self.observation_space, self.action_space = self.case.get_spaces(n_historical_events)

        if rank:
            self.i_reset = rank
        else:
            self.i_reset = 0


    def generate_expert_trajectories(self, out_dir, eval=False, seed_expert=None, n_experts=None, n_demos_per_expert=None, n_expert_time_steps=None):
        if n_experts is None: n_experts = self.n_experts
        if seed_expert is None: seed_expert = self.seed_expert
        if n_demos_per_expert is None: n_demos_per_expert = self.n_demos_per_expert
        if n_expert_time_steps is None: n_expert_time_steps = self.n_expert_time_steps

        states = []
        actions = []
        sex = []
        age = []

        experts = [1, 5] if (isinstance(self.case, Case24) or isinstance(self.case, Case31)) else range(n_experts)

        for i_expert in experts:
            self.model.spawn_new_customer(i_expert) if seed_expert else self.model.spawn_new_customer()
            
            sex.append(self.model.sex)
            age.append(self.model.age)

            if isinstance(self.case, Case7):
                assert n_expert_time_steps % self.case.N == 0
                sample = self.case.get_sample(self.case.N, self.n_historical_events, int(n_expert_time_steps / self.case.N))
                baskets = np.random.permutation(self.case.adam_baskets[i_expert])
            else:
                sample = self.case.get_sample(n_demos_per_expert, self.n_historical_events, n_expert_time_steps)

            temp_states = []
            temp_actions = []

            for j, subsample in enumerate(sample):
                history = subsample[0]
                data = subsample[1]

                if isinstance(self.case, Case7):
                    # adam = random.sample(self.case.adam_baskets[i_expert], 1)
                    adam = baskets[j]
                    initial_state = self.case.get_initial_state(history, adam)
                else:
                    initial_state = self.case.get_initial_state(history, i_expert)

                self.state = initial_state

                temp_states.append(initial_state)

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

        if out_dir is not None:
            # Save trajectories
            if eval:
                np.savez(out_dir + '/eval_expert_trajectories.npz', 
                    states=np.array(states, dtype=object),
                    actions=np.array(actions, dtype=object)
                )
            else:
                np.savez(out_dir + '/expert_trajectories.npz', 
                    states=np.array(states, dtype=object), 
                    actions=np.array(actions, dtype=object)
                )

        return {'states': states, 'actions': actions, 'sex': sex, 'age': age}


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

        if self.seed_agent:
            assert self.seed_expert, 'It only makes sense to seed agent if expert(s) are seeded'
            if isinstance(self.case, Case24) or isinstance(self.case, Case31):
                # Choose between Expert 2 and Expert 6
                seed = np.random.choice([1, 5])
            elif isinstance(self.case, Case221) or isinstance(self.case, Case222) or isinstance(self.case, Case7) or isinstance(self.case, Case23):
                seed = self.i_reset % self.n_experts
                if self.n_processes:
                    self.i_reset += self.n_processes
                else:
                    self.i_reset += 1
            else:
                seed = np.random.randint(0, self.n_experts)
        else:
            seed = None

        self.model.spawn_new_customer(seed)

        # Sample expert trajectory
        if isinstance(self.case, Case31):
            # Using days between purchases -> cannot choose sequence of fixed length
            sample = self.case.get_sample(
                n_demos_per_expert=10, 
                n_historical_events=self.n_historical_events, 
                n_time_steps=self.n_expert_time_steps
                )

            i = np.random.randint(0, 10)

            history, _ = sample[i]
        else:
            sample = self.case.get_sample(
                n_demos_per_expert=1, 
                n_historical_events=self.n_historical_events, 
                n_time_steps=self.n_expert_time_steps
                )

            history, data = sample[0]

            all_data = np.hstack((history, data))
            _, n = all_data.shape

            i = np.random.randint(0, n-self.n_historical_events)

            history = all_data[:, i:i+self.n_historical_events]

        if isinstance(self.case, Case7):
            adam = random.sample(self.case.adam_baskets[seed], 1)[0]
            self.state = self.case.get_initial_state(history, adam)
        else:
            self.state = self.case.get_initial_state(history, seed)

        self.n_time_steps = 0
        
        return np.array(self.state)
    

    def render(self, mode='human', close=False):
        pass


###########################
########## Trash ##########
###########################

'''
temp = int((self.n_historical_events + self.n_expert_time_steps) / self.n_historical_events)

        sample = self.case.get_sample(temp, self.n_historical_events, 0)
        # sample is an array of tuples (history, data) of length n_demos_per_expert
        # choose a random history
        i = np.random.randint(0, temp)
        history, _ = sample[i]

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
