from numpy import load

data = load('expert_trajectories.npz', allow_pickle=True)
lst = data.files
for item in lst:
    #print(item)
    print(data[item])