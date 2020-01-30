import dgm as dgm
import timeseriesanalysis as timeseriesanalysis
import numpy as np
import pandas as pd
from dtw import dtw,accelerated_dtw
import matplotlib.pyplot as plt



time_steps = 10
n_experts = 1
data = np.zeros((n_experts*6, time_steps))
t = range(time_steps)
model = dgm.DGM()

for i in range(n_experts):
    model.spawn_new_customer()
    data[i*6:i*6+6,:] = model.sample(time_steps)
    print("Age:    ", model.age.transpose())
    print("Sex:    ", model.sex)
#print("Alpha0: ", model.alpha0.transpose())
#print("Alpha1: ", model.alpha1.transpose())
#print("Beta0:  ", model.beta0)
#print("Beta1:   ", model.beta1.transpose())
#print("Gamma:  ", model.gamma.transpose())
#print("Lambda: ", model.lambd.transpose())

#print(X.transpose())

#fig, ax = plt.subplots(nrows=3, ncols=2)
#i = 0
#for row in ax:
#   for col in row:
       #n, bins, patches = col.hist(X[i,:])
       #col.plot(t,inventory[i,:])
#       col.plot(t,X[i,:])

#       i += 1
#plt.show()

#TS1 = data[0,:]
#np.savetxt('test.csv', TS1, delimiter=',')
#TS2 = data[6,:]

###--- Dynamic time wrapping ---###
#from tslearn.metrics import dtw as multiVariateDTW
# sim_meas = np.zeros((n_experts, n_experts))
# for i in range(n_experts):
#     for j in range(n_experts):
#         sim_meas[i,j] = multiVariateDTW(data[i*6:i*6+6,:], data[j*6:j*6+6,:])

# print(sim_meas)
# distArray = ssd.squareform(sim_meas)

###--- Extract features using tsfresh ---###
# from tsfresh import extract_features, select_features
# from tsfresh.utilities.dataframe_functions import impute

# print(data)
# TS = pd.DataFrame({'PG1': data[0, :], 'PG2': data[1, :], 'PG3': data[2, :], 'PG4': data[3, :], 'PG5': data[4, :], 'PG6': data[5, :]})
# TS['id'] = 0
# print(TS)
# extracted_features = extract_features(TS, column_id='id')
# print(extracted_features)
# impute(extracted_features)
# print(extracted_features)
# print(TS.shape)
# print(np.zeros(time_steps).shape)
# print(len(TS))
# print(len(np.zeros(time_steps)))
# features_filtered = select_features(extracted_features, np.zeros(time_steps))
#print(features_filtered)


###--- Hierachical clustering ---###
# import scipy.spatial.distance as ssd
# from scipy.cluster.hierarchy import dendrogram, linkage

# linked = linkage(distArray, 'single')

# labelList = range(1, 11)

# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='top',
#             labels=labelList,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()

###--- TS analysis for univariate TS using autocorrelation etc ---###
# TSanalysis = timeseriesanalysis.TimeSeriesAnalysis()
# autocorrTS1 = TSanalysis.get_autocorr(TS1, length=int(time_steps*0.8))
# autocorrTS2 = TSanalysis.get_autocorr(TS2, length = int(time_steps*0.8))
# crosscorr = TSanalysis.get_crosscorr(autocorrTS1, autocorrTS2)
# d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(autocorrTS1,autocorrTS2, dist='euclidean')
# Pearson = np.corrcoef(autocorrTS1, autocorrTS2)


# plt.subplot(2,3,1)
# plt.title('Time series, Pearson coeff = %f' %Pearson[1,0])
# plt.plot(t, TS1)
# plt.plot(t, TS2)
# plt.subplot(2,2,2)
# plt.title('Auto correlation')
# plt.plot(range(int(time_steps*0.8)), autocorrTS1)
# plt.plot(range(int(time_steps*0.8)), autocorrTS2)
# plt.subplot(2,2,3)
# plt.title('Cross correlation')
# plt.xcorr(autocorrTS1, autocorrTS2)
# plt.subplot(2,2,4)
# plt.title('Dynamic time warping')
# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.xlabel('Subject1')
# plt.ylabel('Subject2')
# plt.title(f'DTW Minimum Path with minimum distance: {np.round(d,2)}')
# plt.show()





# Augmented Dickey-Fuller test: https://machinelearningmastery.com/time-series-data-stationary-python/
# from statsmodels.tsa.stattools import adfuller
#for i in range(inventory.shape[0]):
#    result = adfuller(X[i,:])
#    print('ADF Statistic: %f' % result[0])
#    print('p-value: %f' % result[1])
#    print('Critical Values:')
#    for key, value in result[4].items():
#        print('\t%s: %.3f' % (key, value))
