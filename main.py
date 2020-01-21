import dgm as dgm
import numpy as np

model = dgm.DGM()
model.spawn_new_customer()
X = model.sample(30)
print("Age:    ", model.age.transpose())
print("Sex:    ", model.sex)
print("Alpha0: ", model.alpha0.transpose())
print("Alpha1: ", model.alpha1.transpose())
print("Beta0:  ", model.beta0)
print("Beta1:   ", model.beta1.transpose())
print("Gamma:  ", model.gamma.transpose())
print("Lambda: ", model.lambd.transpose())

print(X.transpose())