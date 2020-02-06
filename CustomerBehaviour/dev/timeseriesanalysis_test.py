#from CustomerBehaviour.tools.generate_data import User
import os
import sys
print('path:')
print(sys.path)

print(os.getcwd())
sys.path.append(os.getcwd())
#sys.path.append(os.getcwd() + '/CustomerBehaviour/tools')
print('path:')
print(sys.path)

#from ..tools.generate_data import User
from CustomerBehaviour.tools.generate_data import User
from CustomerBehaviour.tools.timeseriesanalysis import TimeSeriesAnalysis

usr = User(time_steps = 30)
print(usr.time_series)
print(usr.time_series_discrete)
print(usr.discrete_buying_events)
print('Sex: ' + str(usr.sex))
print('Age: ' + str(usr.age))


TS = TimeSeriesAnalysis(usr.time_series)
TS.get_features()
print(TS.mean_freq)

# print('mean freq: ' + str(usr.mean_freq))
# print('std freq: ' + str(usr.std_freq))
# print('mean cost: ' + str(usr.mean_cost))
# print('std cost: ' + str(usr.std_cost))


# plt.legend(legensd)
# plt.show()

# ## --- plot dependent on sex --- ###
# n_experts = 100
# costs = list()
# freqs = list()
# color = list()
# for i in range(n_experts):
#     usr = User(model=dgm, time_steps=300)
#     costs.append(usr.mean_cost)
#     freqs.append(usr.mean_freq)
#     color.append(usr.sex_color)

# plt.scatter(costs, freqs, c = color)
# plt.show()