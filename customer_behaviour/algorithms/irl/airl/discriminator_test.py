import chainer
import numpy as np
import chainer.functions as F
#from customer_behaviour.algorithms.irl.common.model import MLP
from chainer.link_hooks.spectral_normalization import SpectralNormalization
#from chainerrl.links.mlp import MLP
from chainer.initializers import LeCunNormal
from chainer import links as L

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MLP(chainer.Chain):

    def __init__(self, in_size, n_units, n_out = 1):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, in_size)  # n_in -> in_size
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, 1)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class Discriminator():
    def __init__(self, in_size, n_units = 32, out_size = 1, gpu=-1):
        self.discriminator_net = MLP(in_size, n_units, out_size)
        if gpu >= 0:
            self.discriminator_net.to_gpu(gpu)

    #def __init__(self, in_size, hidden_sizes = [64, 64], out_size = 1, gpu=-1):
        #self.discriminator_net = MLP(2, 32, 1)

        self.discriminator_optimizer = chainer.optimizers.Adam()
        self.discriminator_optimizer.setup(self.discriminator_net)

    def __call__(self, x):
        return F.sigmoid(self.discriminator_net(x))

    def train(self, states, actions, action_probs, targets, xp):
        """
        Return the loss function of the discriminator to be optimized.
        As in discriminator, we only want to discriminate the expert from
        learner, thus this is a binary classification problem.
        Unlike Discriminator used for GAIL, the discriminator in this class
        take a specific form, where
                        exp{f(s, a)}
        D(s,a) = -------------------------
                  exp{f(s, a)} + \pi(a|s)
        """
        state_action = []
        for state, action in zip(states, actions):
            array = np.append(state, float(action))

            state_action.append(array.reshape((-1,1)))

        with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
            log_p_tau = self.get_rewards(xp.asarray(np.concatenate([s_a.T.astype('float32') for s_a in state_action]))).data


        # Concatenate the log p(\tau) and log q(\tau) to compute sum
        # After concatenation, log_pq should have size N * 2
        log_pq = np.concatenate((log_p_tau, action_probs.reshape((-1,1))), axis = 1)

        # Since log_pq is of size N * 2, we should sum each row, thus
        # the answer is of size N * 1
        log_pq = F.logsumexp(log_pq, axis = 1).data


        # Binary Cross Entropy Loss
        # loss_n= - [y_n * log(p(x_n)) + (1 − y_n) * log(1 − p(x_n))]
        # where x_n is the input, and y_n is the output

        ones = np.ones_like(targets)

        total_loss = (targets * (log_p_tau - log_pq.reshape((-1,1))) + (ones - targets) * (action_probs.reshape((-1,1)) - log_pq))
        loss = -F.mean(total_loss)

        self.discriminator_net.cleargrads()
        loss.backward()
        self.discriminator_optimizer.update()
        return loss

    def get_rewards(self, state_action_pair):
        return self.discriminator_net(state_action_pair) #should the reward be the output from sigmoid or raw output?
