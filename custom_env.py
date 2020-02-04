import gym
from gym import spaces

N_DISCRETE_ACTIONS = 2
N_HISTORICAL_EVENTS = 20
N_FIXED_FEATURES = 2

class DiscreteBuyingEvents(gym.Env):
  """Custom Environment that follows gym interface"""
  #metadata = {'render.modes': ['human']}

  def __init__(self, historic_data):
    super(DiscreteBuyingEvents, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.historic_data
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
    # Example for using image as input:
    self.observation_space = spaces.Discrete(N_HISTORICAL_EVENTS+N_FIXED_FEATURES)

  def step(self, action):
    # Execute one time step within the environment
    if action == 0:
        # Did not buy
        df2 = pd.DataFrame([0], columns=list(''))
        df.append(df2)

  def reset(self):
    # Reset the state of the environment to an initial state
    
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    