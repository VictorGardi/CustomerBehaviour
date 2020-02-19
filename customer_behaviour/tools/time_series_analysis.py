import numpy as np

class FeatureExtraction():
    def __init__(self, x, case='discrete_events'):
        if x.ndim > 1:
            self.x = x
        else:
            self.x = np.reshape(x, (1, x.size))
        self.n_product_groups = self.x.shape[0]
        self.case = case

    def get_features(self):
        feature_vector = []
        if self.case == 'discrete_events':
            mean_purchase_freqs, std_purchase_freqs = self.get_mean_std_purchase_frequency()
            min_elapsed_days, max_elapsed_days = self.get_min_max_elapsed_days_between_purchases()

            for i in range(self.n_product_groups):
                feature_vector.append(mean_purchase_freqs[i])
                feature_vector.append(std_purchase_freqs[i])
                feature_vector.append(min_elapsed_days[i])
                feature_vector.append(max_elapsed_days[i])
        else:
            raise NotImplementedError

        return feature_vector

    def extract_general_features():
        pass

    def get_min_max_elapsed_days_between_purchases(self):
        '''
        Returns min and max elapsed days between two purchases
        '''
        min_elapsed_days = list()
        max_elapsed_days = list()
        for i in range(self.n_product_groups):
            tmp_list = list(self.x[i, :])
            if tmp_list.count(tmp_list[0]) == len(tmp_list):
                # All elements are the same
                if tmp_list[0] == 0:
                    # The customer never buys
                    min_elapsed_days.append(len(tmp_list))
                    max_elapsed_days.append(len(tmp_list))
                elif tmp_list[0] == 1:
                    # The customer always buys
                    min_elapsed_days.append(1)
                    max_elapsed_days.append(1)
                else:
                    raise NotImplementedError
                continue
            indices = np.argwhere(tmp_list)
            tmp = list(np.diff([x[0] for x in indices]))
            if not tmp:
                # only one peak...
                # TODO: make better solution 
                min_elapsed_days.append(len(tmp_list))
                max_elapsed_days.append(len(tmp_list))
            # tmp.append(indices[0][0] + 1)  # the last entry in the history was a purchase
            else:
                min_elapsed_days.append(min(tmp))
                max_elapsed_days.append(max(tmp))
        return min_elapsed_days, max_elapsed_days

    def get_mean_std_purchase_frequency(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the time series
        indices = np.argwhere(self.x)  # indices is a list of lists where each sublist is the index of a non-zero element (e.g. [i, j] if self.x is a matrix)
        for i in range(self.n_product_groups):
            # Calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i), 1])
            if tmp.size != 0:
                mean_frequencies.append(np.mean(tmp))
                std_frequencies.append(np.std(tmp))
            else:
                # The customer never buys (there are no non-zero elements in self.x)
                mean_frequencies.append(self.x.shape[1])
                std_frequencies.append(self.x.shape[1])
        return mean_frequencies, std_frequencies


###########################
########## Trash ##########
###########################

'''
class TimeSeriesAnalysis():
    def __init__(self, x, case = 'discrete_case'):
        """Both x and y are time series"""
        if x.ndim > 1:
            self.x = x
        else:
            self.x = np.reshape(x, (1, x.size))
        self.n_product_groups = self.x.shape[0]

    def get_features(self):
        self.mean_freq, self.std_freq = self.get_mean_std_freq()
        self.mean_cost, self.std_cost = self.get_mean_std_cost()
        return [self.mean_freq, self.std_freq]

    def get_min_max_elapsed_days(self):
        # Days between purchases
        min_elapsed_days = list()
        max_elapsed_days = list()
        for i in range(self.n_product_groups):
            tmp_list = list(self.x[i, :])
            if tmp_list.count(tmp_list[0]) == len(tmp_list):
                # All elements are the same
                if tmp_list[0] == 0:
                    # The customer never buys
                    min_elapsed_days.append(len(tmp_list))
                    max_elapsed_days.append(len(tmp_list))
                elif tmp_list[0] == 1:
                    # The customer always buys
                    min_elapsed_days.append(1)
                    max_elapsed_days.append(1)
                else:
                    raise NotImplementedError
                continue
            indices = np.argwhere(tmp_list)
            tmp = list(np.diff([x[0] for x in indices]))
            tmp.append(indices[0][0] + 1)  # the last entry in the history was a purchase
            min_elapsed_days.append(min(tmp))
            max_elapsed_days.append(max(tmp))
        return min_elapsed_days, max_elapsed_days


    def get_mean_std_freq(self):
        mean_frequencies = list()
        std_frequencies = list()
        # Find the indices of non-zero values in the time series
        indices = np.argwhere(self.x)  # indices is a list of lists where each sublist is the index of a non-zero element (e.g. [i, j] if self.x is a matrix)
        for i in range(self.n_product_groups):
            # Calculate the distance between non-zero values
            tmp = np.diff(indices[np.where(indices[:,0] == i), 1])
            if tmp.size != 0:
                mean_frequencies.append(np.mean(tmp))
                std_frequencies.append(np.std(tmp))
            else:
                mean_frequencies.append(0)
                std_frequencies.append(0)
        return mean_frequencies, std_frequencies

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
        tmp = [1] + [np.corrcoef(self.x[:-i], self.x[i:])[0,1] for i in range(1, shift)]
        return np.array(tmp
'''
        