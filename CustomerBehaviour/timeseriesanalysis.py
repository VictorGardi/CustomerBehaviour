import numpy as np

class TimeSeriesAnalysis:
    def __init__(self):
        pass

    def get_autocorr(self, x, length = 20):
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
        return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
            for i in range(1, length)])

    def get_crosscorr(self, x, y):
        return np.correlate(x, y,"full")
        