import chainer
import numpy as np
import collections
import chainer.functions as F
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents import PPO, TRPO
from chainerrl.policies import SoftmaxPolicy
from itertools import chain
from customer_behaviour.algorithms.irl.common.utils.mean_or_nan import mean_or_nan
from custom_gym.envs.discrete_buying_events import Case22, Case23


class GAIL(PPO):
    def __init__(self, env, discriminator, demonstrations, noise = 0.1, gamma=None, PAC_k=1, discriminator_loss_stats_window=1000, **kwargs):
        # super take arguments for dynamic inheritance
        super(self.__class__, self).__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.PAC_k = PAC_k
        self.noise = noise

        self.discriminator = discriminator

        self.demo_states = self.xp.asarray(np.asarray(list(chain(*demonstrations['states']))).astype(np.float32))
        self.demo_actions = self.xp.asarray(np.asarray(list(chain(*demonstrations['actions']))).astype(np.float32))

        self.expert_ratio = [get_purchase_ratio(i) for i in demonstrations['actions']]

        self.discriminator_loss_record = collections.deque(maxlen=discriminator_loss_stats_window)
        self.D_output_mean = collections.deque(maxlen=discriminator_loss_stats_window)
        self.mod_rewards = collections.deque(maxlen=discriminator_loss_stats_window)
        self.rewards = collections.deque(maxlen=discriminator_loss_stats_window)
        
    def _update(self, dataset):
        # override func
        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        xp = self.xp

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size, shuffle=True)
        loss_mean = 0
        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = self.batch_states([b['state'] for b in batch], xp, self.phi)
            actions = xp.array([b['action'] for b in batch])

            self.demonstrations_indexes = np.random.permutation(len(self.demo_states))[:len(states)]

            demo_states, demo_actions = [d[self.demonstrations_indexes] for d in (self.demo_states, self.demo_actions)]

            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
                demo_states = self.obs_normalizer(demo_states, update=False)


            self.discriminator.train(self.convert_data_to_feed_discriminator(demo_states, demo_actions),
                                     self.convert_data_to_feed_discriminator(states, actions))
            loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)

            self.discriminator_loss_record.append(float(loss_mean.array))
        super(self.__class__, self)._update(dataset)

    def _update_if_dataset_is_ready(self):
        # override func
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else
               sum(len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            # update reward in self.memory
            self._flush_last_episode()
            transitions = list(chain.from_iterable(self.memory))
            states = self.xp.asarray(np.concatenate([transition['state'][None] for transition in transitions]))
            actions = self.xp.asarray(np.concatenate([transition['action'][None] for transition in transitions]))
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                D_outputs = self.discriminator.get_rewards(self.convert_data_to_feed_discriminator(states, actions, flag='reward')).array
            
            self.D_output_mean.append(float(np.mean(D_outputs)))
            
            s_a = np.concatenate((states, actions.reshape((-1,1))), axis=1)

            # mod_reward = lambda*reward-(1-lambda)*(get_purchase_ratio(expert) - get_purchase_ratio(agent))
            mod_rewards_temp = []
            rewards_temp = []
            i = 0
            for episode in self.memory:
                for transition in episode:
                    if self.gamma and isinstance(self.env.case, Case22) or isinstance(self.env.case, Case23):              
                        # get which expert a s_a pair belongs to from dummy variables which are placed in beginning of each s_a pair          
                        exp_idx = s_a[i,:self.env.n_experts].tolist().index(1)
                        mod_reward = self.gamma*abs((self.expert_ratio[exp_idx] - get_purchase_ratio(s_a[i,self.env.n_experts:])))
                        transition['reward'] = float(D_outputs[i])-mod_reward
                        mod_rewards_temp.append(mod_reward)
                        rewards_temp.append(transition['reward'])
                    else:
                        transition['reward'] = float(D_outputs[i])
                    
                    i += 1
            dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                )
            #dataset = self._make_dataset()
            assert len(dataset) == dataset_size
            self._update(dataset)
            self.memory = []
            
            self.mod_rewards.append(float(np.mean(mod_rewards_temp)))
            self.rewards.append(float(np.mean(rewards_temp)))

    def convert_data_to_feed_discriminator(self, states, actions, flag='loss'):

        xp = self.model.xp

        #if isinstance(self.model.pi, SoftmaxPolicy):
            # if discrete action
        #    actions = xp.eye(self.model.pi.model.out_size, dtype=xp.float32)[actions.astype(xp.int32)]
        if self.noise:
            states = states.astype(xp.float32)
            states += xp.random.normal(loc=0., scale=self.noise, size=states.shape)
            actions = actions.astype(xp.float32)
            actions += xp.random.normal(loc=0., scale=self.noise, size=actions.shape)
        
        if isinstance(self.env.case, Case22) or isinstance(self.env.case, Case23):
            # Do not show dummy encoding to discriminator
            states = [s[self.env.n_experts:] for s in states]

        if self.PAC_k > 1: #PACGAIL
            # merge state and actions into s-a pairs
            s_a = np.concatenate((xp.array(states), xp.array(actions).reshape((-1,1))), axis=1) #4096*101

            # if reward --> get rewards for the same sequence appended together 
            if flag == 'reward':
                s_a = s_a.tolist()
                stacked_sa = [i*self.PAC_k for i in s_a]
                
                return chainer.Variable(np.asarray(stacked_sa, dtype=np.float32))
            else:
                #Update D
                n_sa = s_a.shape[0]
                assert n_sa % self.PAC_k == 0
                stacks = np.split(s_a, int(n_sa/self.PAC_k), axis=0)
                stacked = [np.concatenate(i) for i in stacks]

                return chainer.Variable(np.asarray(stacked, dtype=np.float32))
        return F.concat((xp.array(states), xp.array(actions, dtype=np.float32).reshape((-1,1))))
    
    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_D_output', mean_or_nan(self.D_output_mean)), 
                ('average_mod_rewards', mean_or_nan(self.mod_rewards)),
                ('average_rewards', mean_or_nan(self.rewards))] + super().get_statistics()


def gailtype_constructor(rl_algo=TRPO):
    _gail_parent = GAIL.mro()[1]
    _gail_func_dict = {func: getattr(GAIL, func) for func in dir(GAIL) if callable(getattr(GAIL, func))
                       and (not func.startswith("__") or func == '__init__')
                       and (not hasattr(_gail_parent, func)
                            or not getattr(GAIL, func) == getattr(_gail_parent, func))}
    return type("GAIL" + rl_algo.__name__.upper(), (rl_algo,), _gail_func_dict)


def get_purchase_ratio(sequence):
    return np.sum(sequence)/len(sequence)


# GAILTRPO do not work because TRPO's interface is not compatible with PPO
#GAILTRPO = gailtype_constructor(rl_algo=TRPO)
