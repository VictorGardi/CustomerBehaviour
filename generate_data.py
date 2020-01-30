import dgm as dgm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        if self.sex == 1:
            self.sex_color = 'red'
        elif self.sex == 0:
            self.sex_color = 'blue'

        if self.age < 30:
            self.age_color = 'red'
            self.value = 0
        elif 30 <= self.age < 40:
            self.age_color = 'blue'
            self.value = 1
        elif 40 <= self.age < 50:
            self.age_color = 'green'
            self.value = 2
        elif 50 <= self.age < 60:
            self.age_color = 'yellow'
            self.value = 3
        elif 60 <= self.age < 70:
            self.age_color = 'purple'
            self.value = 4
        elif 70 <= self.age:
            self.age_color = 'black'
            self.value = 5
        

    def get_discrete_receipt(self):
        time_series = self.model.sample(self.time_steps)
        time_series[time_series > 0] = 1
        return time_series

    def get_features(self):
        self.mean_freq, self.std_freq = self.get_mean_std_freq()
        self.mean_cost, self.std_cost = self.get_mean_std_cost()

    def get_mean_std_freq(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the discrete time series
        indices = np.argwhere(self.time_series)
        for i in range(self.n_product_groups):
            # calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i),1])
            mean_frequencies.append(np.mean(tmp))
            std_frequencies.append(np.std(tmp))
        return np.mean(mean_frequencies), np.std(std_frequencies)

    def get_mean_std_cost(self):
        costs = list()
        indices = np.nonzero(self.time_series)
        costs = self.time_series[indices]
        return np.mean(costs), np.std(costs)


# usr = User(model = dgm, time_steps = 80)
# print(usr.time_series)
# print('Sex: ' + str(usr.sex))
# print('Age: ' + str(usr.age))
# usr.get_features()
# print('mean freq: ' + str(usr.mean_freq))
# print('std freq: ' + str(usr.std_freq))
# print('mean cost: ' + str(usr.mean_cost))
# print('std cost: ' + str(usr.std_cost))

n_experts = 1000
costs = list()
freqs = list()
colors = ('red', 'green', 'blue', 'yellow', 'black', 'purple')
values = list()
legends = ('18-29', '30-39', '40-49', '50-59', '60-69', '70-80')
sex_colors = ('red', 'blue')
sex_values = list()
sex_legends = ('Woman', 'Man')

for i in range(n_experts):
    usr = User(model=dgm, time_steps=300)
    costs.append(usr.mean_cost)
    freqs.append(usr.mean_freq)
    values.append(usr.value)
    sex_values.append(usr.sex)

plt.figure(1)
scatter = plt.scatter(costs, freqs, c = values, cmap = ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=legends)
plt.figure(2)
scatter1 = plt.scatter(costs, freqs, c = sex_values, cmap = ListedColormap(sex_colors))
plt.legend(handles=scatter1.legend_elements()[0], labels=sex_legends)
#plt.legend(legensd)
plt.show()

### --- plot dependent on sex --- ###
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









