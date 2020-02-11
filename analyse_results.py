from customer_behaviour.tools.visualization import Result
import os

expert_data = os.getcwd() + '/results/gail/discrete_events/1_expert(s)/1_product(s)/2020-02-11_12-46-53/expert_trajectories.npz'
agent_data = os.getcwd() + '/results/gail/discrete_events/1_expert(s)/1_product(s)/2020-02-11_12-46-53/trajectories.npz'

result = Result(expert_data, agent_data)

result.plot_uni_time_series()
#result.plot_univariate_time_series()