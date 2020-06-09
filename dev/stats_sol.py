import numpy as np
import matplotlib.pyplot as plt
from customer_behaviour.tools.dgm import DGM

dgm = DGM()

sample_length = 365
n_experts = 1000
samples = []
discrete_samples = []

def get_purchase_ratio(sequence):
    return np.count_nonzero(sequence)/len(sequence)
ratios = []
for _ in range(n_experts):
    dgm.spawn_new_customer()
    sample = np.sum(dgm.sample(sample_length), axis=0)
    #samples.append(sample)
    #sample[sample > 0] = 1
    #discrete_samples.append(sample)
    ratios.append(get_purchase_ratio(sample))


#ratios = [get_purchase_ratio(sample) for sample in samples]
#print(ratios)
print(np.mean(ratios))

plt.hist(ratios, density=True, bins=20)
plt.ylabel('Probability')
plt.xlabel('Data')
plt.show()

def get_event_hist(samples):
    pass


    

    