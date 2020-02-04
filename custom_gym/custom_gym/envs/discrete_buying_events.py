import gym
from gym import spaces


N_ACTIONS = 2  # do not buy something (0) & buy something (1)
N_AGE_DIVISIONS = 6  # 18-29 (0) & 30-39 (1) & 40-49 (2) & 50-59 (3) & 60-69 (4) & 70-80 (5) 
N_HISTORICAL_EVENTS = 20
N_FIXED_FEATURES = 2
N_MAX_TIME_STEPS = 1000


class DiscreteBuyingEvents(gym.Env):
  """Custom Environment that follows gym interface"""
  # metadata = {'render.modes': ['human']}

  def __init__(self, sex, age, history):
    super(DiscreteBuyingEvents, self).__init__()
    
    assert sex in {0, 1}
    assert age in {0, 1, 2, 3, 4, 5}
    assert len(history) == N_HISTORICAL_EVENTS and min(history) >= 0 and max(history) <= 1

    self.sex = sex
    self.age = age
    self.history = history

    self.n_time_steps = 0

    self.state = (self.sex, self.age, self.history)

    # Define action and observation space
    self.action_space = spaces.Discrete(N_ACTIONS)
    
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(N_AGE_DIVISIONS), spaces.MultiBinary(N_HISTORICAL_EVENTS)))

    # self.observation_space = spaces.Dict({"sex": spaces.Discrete(2), "age": spaces.Discrete(N_AGE_DIVISIONS),
      # "history": spaces.MultiBinary(N_HISTORICAL_EVENTS)})
    
    # print(self.observation_space.sample())

  def step(self, action):
    # Execute one time step within the environment

    assert action == 0 or action == 1

    history = self.state[2]
    new_history = [*history[1:], action]
    self.state[2] = new_history

    self.n_time_steps += 1

    done = self.n_time_steps > N_MAX_TIME_STEPS

    reward = 0

    return self.state, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    self.state = (self.sex, self.age, self.history)
    return self.state
    
  def render(self, mode='human', close=False):
    pass
    

# DiscreteBuyingEvents(0, 0, N_HISTORICAL_EVENTS * [0])


# class FullReceipt(gym.Env):
 # def __init__(self):
  # self.observation_space = spaces.Dict({"sex": spaces.Discrete(2), "age": spaces.Discrete(3)})


