import numpy as np

class TimeSeriesAnalysis:
    def __init__(self, x, y = None):
        """Both x and y are time series"""
        self.x = x
        self.y = y
        self.n_product_groups = self.x.shape[0]

    def get_features(self):
        self.mean_freq, self.std_freq = self.get_mean_std_freq()
        self.mean_cost, self.std_cost = self.get_mean_std_cost()

    def get_mean_std_freq(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the time series
        indices = np.argwhere(self.x)
        for i in range(self.n_product_groups):
            # calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i),1])
            mean_frequencies.append(np.mean(tmp))
            std_frequencies.append(np.std(tmp))
        return np.mean(mean_frequencies), np.std(std_frequencies)

    def get_mean_std_cost(self):
        costs = list()
        indices = np.nonzero(self.x)
        costs = self.x[indices]
        return np.mean(costs), np.std(costs)

    def get_autocorr(self, shift = 20):
        """ Get autocorrelation of time series x. 
        The autocorrelation quantifies the average similarity between 
        the signal and a shifted version of the same signal, as a 
        function of the delay between the two. In other words, the 
        autocorrelation can give us information about repeating patterns 
        as well as the timescale of the signal's fluctuations. The faster 
        the autocorrelation decays to zero, the faster the signal varies.
        """
        # self.x = x
        # n = len(self.x)
        # variance = self.x.var()
        # x = self.x-self.x.mean()
        # r = np.correlate(x, x, mode = 'full')[-n:]
        # #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        # result = r/(variance*(np.arange(n, 0, -1)))
        # return result
        #result = np.correlate(x, x, mode='full')
        #return result[result.size/2:]
        return np.array([1]+[np.corrcoef(self.x[:-i], self.x[i:])[0,1]  \
            for i in range(1, shift)])

    def get_crosscorr(self):
        return np.correlate(self.x, self.y,"full")
        