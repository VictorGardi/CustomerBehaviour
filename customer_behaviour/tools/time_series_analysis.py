from math import floor
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
            max_purchases_per_week = self.get_max_purchases_per_window(7)
            max_purchases_per_two_weeks = self.get_max_purchases_per_window(14)
            max_purchases_per_month = self.get_max_purchases_per_window(28)

            for i in range(self.n_product_groups):
                feature_vector.append(mean_purchase_freqs[i])
                feature_vector.append(std_purchase_freqs[i])
                feature_vector.append(min_elapsed_days[i])
                feature_vector.append(max_elapsed_days[i])
                feature_vector.append(max_purchases_per_week[i])
                feature_vector.append(max_purchases_per_two_weeks[i])
                feature_vector.append(max_purchases_per_month[i])
                #autocorr = self.get_autocorr(self.x[i,:])
                #feature_vector.extend(autocorr)
        else:
            raise NotImplementedError
        # print(feature_vector)
        return feature_vector


    def get_max_purchases_per_window(self, window_width):
        max_n_purchases_ls = []
        
        for i in range(self.n_product_groups):
            max_n_purchases = 0
            x = self.x[i,:]
            for j in range(len(x) - window_width):
                temp = x[j:j + window_width]
                temp[temp > 0] = 1
                n_purchases = np.sum(temp)
                if n_purchases > max_n_purchases:
                    max_n_purchases = n_purchases
            max_n_purchases_ls.append(max_n_purchases**2)
        return max_n_purchases_ls
                

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
    
    def get_autocorr(self, x, t = 20):
        """ Get autocorrelation of time series x. 
        The autocorrelation quantifies the average similarity between 
        the signal and a shifted version of the same signal, as a 
        function of the delay between the two. In other words, the 
        autocorrelation can give us information about repeating patterns 
        as well as the timescale of the signal's fluctuations. The faster 
        the autocorrelation decays to zero, the faster the signal varies.
        """
        result = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        return result[floor(result.size/2):floor(result.size/2)+t]
        #return np.corrcoef(np.array([x[:-t], x[t:]]))
        #return result[floor(result.size/2):floor(result.size/2) + 32]
        

        