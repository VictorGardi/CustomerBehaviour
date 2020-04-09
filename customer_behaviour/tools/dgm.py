import numpy as np

class DGM:
    
    def __init__(self):
        self.prices = np.reshape(np.array([10,10,6,3,3,2]),(6,1))
        self.alpha0_lower = np.reshape(np.array([5,5.3,5.6,5.9,6.2,6.5]),(6,1))
        self.alpha0_upper = np.reshape(np.array([6,6.3,6.6,6.9,7.2,7.5]),(6,1))
        self.alpha1_upper = np.reshape(np.array([11,11.3,11.6,11.9,12.2,12.5]),(6,1))
        self.alpha1_lower = np.reshape(np.array([8,8.3,8.6,8.9,9.2,9.5]),(6,1))
        self.beta1_lower = np.reshape(np.array([3,3.3,3.6,3.9,4.2,4.5]),(6,1))
        self.beta1_upper = np.reshape(np.array([4,4.3,4.6,4.9,5.2,5.5]),(6,1))
        self.week = np.reshape(np.array([0,0,0,0,0,1,1]),(7,1))
        self.week = self.week / np.sum(self.week)
        
        self.N = len(self.prices)
        self.prices = self.prices / np.max(self.prices)

    def spawn_new_customer(self, seed=None):

        if seed is not None: np.random.seed(seed)

        self.age = np.floor(np.random.uniform(18,80))
        self.sex = np.round(np.random.uniform(0,1)) # Female = 1

        self.alpha0 = np.random.uniform(self.alpha0_lower,self.alpha0_upper)
        self.alpha1 = np.random.uniform(self.alpha1_lower,self.alpha1_upper)
        
        self.beta0 = np.random.uniform(-4,-2)
        self.beta1 = np.random.uniform(self.beta1_lower,self.beta1_upper)
        self.gamma = np.random.triangular(0.1,0.1+0.2*((self.age-18))/62,0.3,(self.N,1)) + np.random.uniform(0,self.sex * 0.1)
        self.lambd = np.random.uniform(1,2,(self.N,1)) 
        self.cost = np.random.triangular(20, 20+580*(self.age-18)/62, 600, (self.N,1))
        
        
        self.prices_dev = np.random.uniform(0,1,(self.N,1))
        self.prices_dev = self.prices_dev / np.sum(self.prices_dev)
        self.prices_customer = 0.6 * self.prices + 0.4 * self.prices_dev
        
        u = np.random.uniform(0,1)
        self.week_dev = np.random.uniform(0,1,(7,1))
        self.week_dev = self.week_dev / np.sum(self.week_dev)
        self.week_customer = u * self.week + (1-u) * self.week_dev
                        
    def _purchase(self, day):
        trigg = self.alpha0 - np.multiply(self.alpha1,self.inventory)
        trigg = np.divide(1,1 + np.exp(-trigg)) 
        #print("Sigma:     ", np.round(100*trigg.transpose()))
        trigg = np.multiply(trigg, self.beta1)/self.N  
        #print("Beta:      ", np.round(100*trigg.transpose()))
        f = self.beta0 + np.sum(trigg) + 2*self.week_customer[day,0] 
        f = np.divide(1,1 + np.exp(-f)) 
        #print("Prob:      ", f)
        return f > np.random.uniform(0,1)
    
    def _update(self, day):
        if self._purchase(day) == True:
            prev_inventory = self.inventory
            self.inventory = np.maximum(self.inventory,np.divide(np.random.gamma(self.lambd),self.lambd))
            purchase = np.floor(np.multiply(self.inventory - prev_inventory, self.prices_customer)*self.cost)
        else:
            self.inventory = np.maximum(0,self.inventory - np.random.uniform(0,self.gamma))
            purchase = np.zeros((self.N,1))
        return purchase

    def sample3(self, n_demos, n_historical_events, n_time_steps):
        self.inventory = np.divide(np.random.gamma(self.lambd), self.lambd)

        sample = []
        l = 0

        for _ in range(n_demos):
            # Sample history for initial state
            history = np.zeros((self.N, 0))
            n_purchases = 0
            while n_purchases < n_historical_events:
                purchase = self._update(day = np.mod(l, 7))
                history = np.column_stack((history, purchase))
                if np.count_nonzero(purchase) > 0:
                    n_purchases += 1
                l += 1

            # Sample data
            data = np.zeros((self.N, n_time_steps))
            for i in range(n_time_steps):
                purchase = self._update(day = np.mod(l, 7))
                data[:, [i]] = purchase
                l += 1
            sample.append((history, data))

        np.random.seed(None)

        return sample


    def sample2(self, n_demos, n_historical_events, n_time_steps, n_product_groups=6):
        self.inventory = np.divide(np.random.gamma(self.lambd),self.lambd)

        all_sample = np.zeros((self.N, 0))

        sample = []

        l = 0

        for _ in range(n_demos):
            # Sample history for initial state
            history = np.zeros((self.N, 0))
            n_buys = np.array(n_product_groups * [0])
            while min(n_buys) < n_historical_events:
                # print(n_buys)
                purchase = self._update(day = np.mod(l,7))
                all_sample = np.column_stack((all_sample, purchase))
                history = np.column_stack((history, purchase))
                temp = [int(x) for x in purchase > 0]
                n_buys += temp[:n_product_groups]
                l += 1
            # Sample data
            data = np.zeros((self.N, n_time_steps))
            for j in range(n_time_steps):
                purchase = self._update(day = np.mod(l,7))
                all_sample = np.column_stack((all_sample, purchase))
                data[:, [j]] = purchase
                l += 1
            sample.append((history, data))
        
        np.random.seed(None)

        '''
        import matplotlib.pyplot as plt
        _, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(all_sample[0, :])
        temp = []
        for s in sample:
            temp.extend(s[0][0, :])
            temp.extend(s[1][0, :])
        ax2.plot(temp)
        plt.show()
        '''
        
        return sample


    def sample(self, L):
        self.inventory = np.divide(np.random.gamma(self.lambd),self.lambd)
        #print("Inventory: ", np.round(100*self.inventory.transpose()))
        sample = np.zeros((self.N,L))   
        inventory = np.zeros((self.N, L))
        for l in range(L):
            purchase = self._update(day = np.mod(l,7))
            #print("")
            #print("Inventory: ", np.round(100*self.inventory.transpose()))
            sample[:,[l]] = purchase
            inventory[:,[l]] = np.round(100*self.inventory)
        np.random.seed(None)  # We do not want the following random values to be affected by the seed used for spawning a new customer
        return sample

    def sample_deterministically(self, L):
        sample = np.zeros((self.N, L))
        for l in range(L):
            if (l+1) % 10 == 0:
                r = np.random.rand()
                sample[:, l] = 1 if r < 0.7 else 0
            else:
                sample[:, l] = 0
        return sample
                    