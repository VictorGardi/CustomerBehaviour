import dgm as dgm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class User:
    def __init__(self, model, time_steps = 50, n_product_groups = 6):
        self.time_steps = time_steps
        self.model = model.DGM()
        self.model.spawn_new_customer()    
        self.age = self.model.age.transpose()
        self.sex = self.model.sex
        self.time_series = self.model.sample(time_steps) 
        self.time_series_discrete = self.get_discrete_receipt()
        self.mean_freq = None
        self.var_freq = None
        self.mean_cost = None
        self.var_cost = None
        self.n_product_groups = n_product_groups
        self.get_features()

    def get_discrete_receipt(self):
        time_series = self.model.sample(self.time_steps)
        time_series[time_series > 0] = 1
        return time_series

    def get_features(self):
        self.mean_freq, self.std_freq = self.get_mean_std_freq()
        self.mean_cost, self.std_cost = self.get_mean_std_cost()
        #self.std_cost = self.get_sdt_cost()

    def get_mean_std_freq(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the discrete time series
        indices = np.argwhere(self.time_series)
        #print(indices)
        for i in range(self.n_product_groups):
            # calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i),1])
            #print(tmp)
            #tmp = np.diff(tmp)
            mean_frequencies.append(np.mean(tmp))
            std_frequencies.append(np.std(tmp))
            #print(mean_frequencies)
            #print(std_frequencies)
        return np.mean(mean_frequencies), np.std(std_frequencies)

    def get_mean_std_cost(self):
        costs = list()
        # Find the indices of non-zero values in the time series
        indices = np.nonzero(self.time_series)
        #print(indices)
        costs = self.time_series[indices]
        #print(costs)
        return np.mean(costs), np.std(costs)

    # def get_plot(self):
    #     plt.scatter(self.mean_freq, self.mean_cost)



# usr = User(model = dgm, time_steps = 80)
# print(usr.time_series)
# print('Sex: ' + str(usr.sex))
# print('Age: ' + str(usr.age))
# usr.get_features()
# print('mean freq: ' + str(usr.mean_freq))
# print('std freq: ' + str(usr.std_freq))
# print('mean cost: ' + str(usr.mean_cost))
# print('std cost: ' + str(usr.std_cost))

n_experts = 200
costs = list()
freqs = list()
color = list()
for i in range(n_experts):
    usr = User(model=dgm, time_steps=100)
    costs.append(usr.mean_cost)
    freqs.append(usr.mean_freq)
    if usr.sex == True:
        color.append('red')
    else:
        color.append('blue')

plt.scatter(costs, freqs, c = color)
plt.show()









