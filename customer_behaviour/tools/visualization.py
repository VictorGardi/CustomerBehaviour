import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


class Result():

	def __init__(self, expert_data, agent_data):

		self.expert_trajectories = self.load_trajectories(expert_data)
		self.agent_trajectories = self.load_trajectories(agent_data)

		self.expert_states, self.expert_actions = self.load_trajecory(expert_data)
		self.agent_states, self.agent_actions = self.load_trajecory(agent_data)

		self.load_trajectories(expert_data)


	def load_data(self, file):
		pass


	def load_trajectories(self, file):
		data = np.load(file, allow_pickle=True)

		assert sorted(data.files) == sorted(['states', 'actions'])

		trajectories = []

		for trajectory_states, trajectory_actions in zip(data['states'], data['actions']):

			state_action_pairs = [(list(state), action) for state, action in zip(trajectory_states, trajectory_actions)]

			trajectories.append(state_action_pairs)

		return trajectories

	def plot_result(self):

		pass
		#expert_sex = self.expert_trajectories[0][]



	def load_trajecory(self, file):
		trajectory = np.load(file, allow_pickle=True)

		assert sorted(trajectory.files) == sorted(['states', 'actions'])

		states = trajectory['states']
		actions = trajectory['actions']

		n_episodes = len(states)



		# print(len(states))

		return states, actions

	def plot_univariate_time_series(self):
		expert_sex = None
		expert_age = None

		fig, (ax1, ax2) = plt.subplots(2, 1)

		n_expert_steps = None

		for i, trajectory in enumerate(self.expert_trajectories):
			temp = []
			
			for j, (state, action) in enumerate(trajectory):
				if j == 0:
					temp.extend(x for x in state[2:])  # state[0] = sex, state[1] = age
					temp.append(action)
				else:
					temp.append(action)
			
			if i == 0: n_expert_steps = len(temp)

			ax1.plot(temp)

		for i, trajectory in enumerate(self.agent_trajectories):

			if i == 100:

				temp = []
				for j, (state, action) in enumerate(trajectory):
					if j == 0:
						temp.extend(x for x in state[2:])  # state[0] = sex, state[1] = age
						temp.append(action)
					else:
						temp.append(action)

					if len(temp) == n_expert_steps:
						break

				ax2.plot(temp)


		plt.show()




def main():
	expert_data = '/Users/antonmatsson/CustomerBehaviour/expert_trajectories.npz'
	agent_data = '/Users/antonmatsson/CustomerBehaviour/deepirl_chainer-master/results/20200206T153935.609555/trajectories.npz'

	result = Result(expert_data, agent_data)

	result.plot_univariate_time_series()


if __name__ == '__main__':
    main()