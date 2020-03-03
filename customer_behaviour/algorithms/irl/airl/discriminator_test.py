import chainer
import numpy as np
import chainer.functions as F
#from customer_behaviour.algorithms.irl.common.model import MLP
from chainer.link_hooks.spectral_normalization import SpectralNormalization
#from chainerrl.links.mlp import MLP
from chainer.initializers import LeCunNormal
from chainer import links as L

from chainerrl import links

class Discriminator():
    def __init__(self, obs_space, action_space, out_size = 1, gpu=-1):
        hidden_sizes = (64,64)
        self.reward_net = links.MLP(obs_space + action_space, out_size, hidden_sizes=hidden_sizes)
        self.value_net = links.MLP(obs_space, out_size, hidden_sizes=hidden_sizes)
        if gpu >= 0:
            self.reward_net.to_gpu(gpu)
            self.value_net.to_gpu(gpu)

        self.reward_optimizer = chainer.optimizers.Adam()
        self.reward_optimizer.setup(self.reward_net)
        self.value_optimizer = chainer.optimizers.Adam()
        self.value_optimizer.setup(self.value_net)

    def __call__(self, x):
        return self.reward_net(x), self.value_net(x)

    def train(self, states, actions, action_logprobs, next_states, targets, xp, gamma = 1):
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

        # Create state-action pairs. Remember that both the agent's and the expert's s-a pairs are in the same list
        state_action = []
        for state, action in zip(states, actions):
            action = np.array([0, 1]) if action == 0 else np.array([0, 1])
            array = np.append(state, action)

            state_action.append(array.reshape((-1,1)))

        # Get rewards for all s-a pairs
        rewards = self.reward_net(xp.asarray([s_a.T.astype('float32') for s_a in state_action])).data
        # Get values for current states
        current_values = self.value_net(xp.asarray([s.T.astype('float32') for s in states])).data
        # get values for next states
        next_values = self.value_net(xp.asarray([s.T.astype('float32') for s in next_states])).data

        # Define log p_tau(a|s) = r + gamma * V(s') - V(s)
        log_p_tau = rewards + gamma*next_values - current_values

        # log_q_tau = logprobs(pi(s)) = logprobs(a) calculated by policy net given a state
        # action_logprobs contains probs for both expert and agent
        log_q_tau = action_logprobs.reshape((-1,1))

        # Concatenate the rewards from discriminator and action probs from policy net to compute sum
        # After concatenation, log_pq should have size N * 2
        log_pq = np.concatenate((log_p_tau, log_q_tau), axis = 1)
        
        # logsumexp = softmax, i.e. for each row we take log(sum(exp(val))) so that we add together probabilities 
        # and then go back to logprobs
        log_pq = F.logsumexp(log_pq, axis = 1).data.reshape((-1,1))

        # Calculate D 
        discrim_output = F.exp(log_p_tau-log_pq)

        # Calculate cross entropy loss

        loss = targets*(log_p_tau-log_pq) + (1-targets)*(log_q_tau-log_pq)

        loss = -F.mean(loss)        

        self.reward_net.cleargrads()
        self.value_net.cleargrads()
        
        self.reward_optimizer.update(loss.backward())
        self.value_optimizer.update(loss.backward())
        return loss       

    def get_rewards(self, state_action_pair):
        return self.reward_net(state_action_pair) #should the reward be the output from sigmoid or raw output?
