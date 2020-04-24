from sklearn.utils import shuffle
import unittest
import chainer
import numpy as np
import collections
import chainer.functions as F
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents import PPO, TRPO
from chainerrl.policies import SoftmaxPolicy
from itertools import chain
from customer_behaviour.algorithms.irl.common.utils.mean_or_nan import mean_or_nan
from custom_gym.envs.discrete_buying_events import Case22, Case23, Case24, Case31, Case221, Case222, Case7, Case4, Case71


class GAIL(PPO):
    def __init__(self, env, discriminator, demonstrations, n_experts, episode_length, adam_days, dummy_D=0, noise = None, gamma=0, PAC_k=1, discriminator_loss_stats_window=1000, **kwargs):
        # super take arguments for dynamic inheritance
        super(self.__class__, self).__init__(**kwargs)

        self.env = env
        self.gamma = gamma
        self.PAC_k = PAC_k
        self.noise = noise
        self.n_experts = n_experts
        self.episode_length = episode_length
        self.dummy_D = dummy_D
        self.adam_days = adam_days

        self.discriminator = discriminator
        
        
        if isinstance(self.env.case, Case221) or isinstance(self.env.case, Case222) or isinstance(self.env.case, Case7) or isinstance(self.env.case, Case23) or isinstance(self.env.case, Case4) \
        or isinstance(self.env.case, Case71):
            self.demo_states = [*demonstrations['states']]
            self.demo_actions = [*demonstrations['actions']]
        else:
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
        #datasets_iter = [chainer.iterators.SerialIterator(
        #    [dataset[i]], self.minibatch_size, shuffle=True) for i in dataset]

        if isinstance(self.env.case, Case221) or isinstance(self.env.case, Case7) or isinstance(self.env.case, Case23) or isinstance(self.env.case, Case4) \
        or isinstance(self.env.case, Case71):
            dataset_states = []
            dataset_actions = []
            for expert in range(self.n_experts):
                min_idx = expert*self.episode_length
                max_idx = (expert + 1)*self.episode_length 
                dataset_states.append([dataset[i]['state'] for i in range(min_idx, max_idx)])
                dataset_actions.append([dataset[i]['action'] for i in range(min_idx, max_idx)])

            loss_mean = 0
            n_mb = int(self.episode_length/self.minibatch_size)
            for epoch in range(self.epochs):

                for expert in range(self.n_experts):
                    demo_states = self.demo_states[expert]
                    demo_actions = self.demo_actions[expert]
                    states = dataset_states[expert]
                    actions = dataset_actions[expert]
                    states, actions = shuffle(np.array(states), np.array(actions))
                    demo_states, demo_actions = shuffle(demo_states, demo_actions)

                    for demo_state, state in zip(demo_states, states):
                    
                        if isinstance(self.env.case, Case7):
                            demo_dummy = list(map(int, list(demo_state[2:self.adam_days+2])))
                            dummy = list(map(int, list(state[2:self.adam_days+2])))

                            if not dummy in self.env.case.adam_baskets[expert]:
                                raise NameError('States are in the wrong order!')
                        elif isinstance(self.env.case, Case71):
                            demo_dummy = list(map(int, list(demo_state[:self.adam_days])))
                            dummy = list(map(int, list(state[:self.adam_days])))

                            if not dummy in self.env.case.adam_baskets[expert]:
                                raise NameError('States are in the wrong order!')
                        else: 
                            demo_dummy = list(map(int, list(demo_state[:self.n_experts])))
                            dummy = list(map(int, list(state[:self.n_experts])))
                            if not demo_dummy == dummy:
                                raise NameError('States are in the wrong order!')
                            else:
                                pass # the order of expert and agent is correct

                    for mb in range(n_mb):
                        min_idx = mb*self.minibatch_size
                        max_idx = (mb + 1)*self.minibatch_size
                        mb_states = states[min_idx:max_idx,:]
                        mb_actions = actions[min_idx:max_idx]
                        if states.shape[0] > demo_states.shape[0]:
                            indices = np.random.choice(demo_states.shape[0], size=self.minibatch_size)
                            mb_demo_states = np.take(demo_states, indices, axis=0)
                            mb_demo_actions = np.take(demo_actions, indices, axis=0)
                        else:
                            mb_demo_states = demo_states[min_idx:max_idx,:]
                            mb_demo_actions = demo_actions[min_idx:max_idx]

                        if self.obs_normalizer:
                            mb_states = self.obs_normalizer(mb_states, update=False)
                            mb_demo_states = self.obs_normalizer(mb_demo_states, update=False)

                        self.discriminator.train(self.convert_data_to_feed_discriminator(mb_demo_states, mb_demo_actions),
                                                self.convert_data_to_feed_discriminator(mb_states, mb_actions))
                        loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)

                        self.discriminator_loss_record.append(float(loss_mean.array))

        elif isinstance(self.env.case, Case222):
            dataset_states = []
            dataset_actions = []
            for expert in range(self.n_experts):
                min_idx = expert*self.episode_length
                max_idx = (expert + 1)*self.episode_length 
                dataset_states.append([dataset[i]['state'] for i in range(min_idx, max_idx)])
                dataset_actions.append([dataset[i]['action'] for i in range(min_idx, max_idx)])

            loss_mean = 0
            n_mb = int(self.episode_length/self.minibatch_size)
            for epoch in range(self.epochs):
                # shuffle states and actions in the same way for each expert's trajectory
                shuffled_agent_states, shuffled_agent_actions = self.shuffle_dataset(dataset_states, dataset_actions)
                shuffled_demo_states, shuffled_demo_actions = self.shuffle_dataset(self.demo_states, self.demo_actions)
                
                for mb in range(n_mb):
                    min_idx = mb*self.minibatch_size
                    max_idx = (mb + 1)*self.minibatch_size
                    
                    for i in range(self.n_experts):
                        expert_states = shuffled_demo_states[i]
                        expert_actions = shuffled_demo_actions[i]
                        agent_states = np.array(shuffled_agent_states[i])
                        agent_actions = np.array(shuffled_agent_actions[i])

                        mb_demo_states = expert_states[min_idx:max_idx,:]
                        mb_demo_actions = expert_actions[min_idx:max_idx]
                        mb_states = agent_states[min_idx:max_idx,:]
                        mb_actions = agent_actions[min_idx:max_idx]
                        
                        if self.obs_normalizer:
                            mb_states = self.obs_normalizer(mb_states, update=False)
                            demo_states = self.obs_normalizer(mb_demo_states, update=False)

                        self.discriminator.train(self.convert_data_to_feed_discriminator(mb_demo_states, mb_demo_actions),
                                                self.convert_data_to_feed_discriminator(mb_states, mb_actions))
                        loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)

                        self.discriminator_loss_record.append(float(loss_mean.array))

        else:
            dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size, shuffle=True)
            loss_mean = 0
            while dataset_iter.epoch < self.epochs:
                batch = dataset_iter.__next__()
                states = self.batch_states([b['state'] for b in batch], xp, self.phi)
                actions = xp.array([b['action'] for b in batch])

                self.demonstrations_indexes = xp.random.permutation(len(self.demo_states))[:len(states)]

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

            mod_rewards_temp = []
            rewards_temp = []
            i = 0
            for episode in self.memory:
                for transition in episode:
                    if self.gamma > 0 and (isinstance(self.env.case, Case22) or isinstance(self.env.case, Case23)):              
                        # get which expert a s_a pair belongs to from dummy variables which are placed in beginning of each s_a pair          
                        exp_idx = s_a[i,:self.env.n_experts].tolist().index(1)
                        mod_reward = self.gamma*abs((self.expert_ratio[exp_idx] - get_purchase_ratio(s_a[i,self.env.n_experts:])))
                        transition['reward'] = float(D_outputs[i])-mod_reward
                        mod_rewards_temp.append(mod_reward)
                        rewards_temp.append(transition['reward'])
                    else:
                        transition['reward'] = float(D_outputs[i])
                        rewards_temp.append(transition['reward'])
                    
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
        if self.noise > 0:
            states = states.astype(xp.float32)
            states += xp.random.normal(loc=0., scale=self.noise, size=states.shape)
            actions = actions.astype(xp.float32)
            actions += xp.random.normal(loc=0., scale=self.noise, size=actions.shape)
        #print('convert_data')
        #print(len(actions))
        #print(len(states))

        if isinstance(self.env.case, Case31):
            temp = []
            for s in states:
                history = s[:-10]
                dummy = s[-10:]

                new_h = []
                for x in history:
                    while x > 1:
                        new_h.append(0)
                        x -= 1
                    new_h.append(1)
                new_h.reverse()

                temp.append(np.concatenate((dummy, new_h[-self.env.n_historical_events:])).astype(xp.float32))
            states = temp
        """
        if isinstance(self.env.case, Case221) or isinstance(self.env.case, Case222) or isinstance(self.env.case, Case7):
            states = [s[self.env.n_experts:] for s in states]
            
        
        if isinstance(self.env.case, Case22) or isinstance(self.env.case, Case23):
            # Do not show dummy encoding to discriminator
            #states = [s[self.env.n_experts:] for s in states]
            pass

        if isinstance(self.env.case, Case24):
            # Do not show dummy encoding to discriminator
            #states = [s[10:] for s in states]
            pass  # Let discriminator see dummy encoding"""

        if not self.dummy_D:
            states = [s[self.env.n_experts:] for s in states]
            if isinstance(self.env.case, Case7):
                states = [s[2+self.adam_days:] for s in states]
            elif isinstance(self.env.case, Case71):
                states = [s[self.adam_days:] for s in states]

        if self.PAC_k > 1: #PACGAIL
            # merge state and actions into s-a pairs
            s_a = xp.concatenate((xp.array(states), xp.array(actions).reshape((-1,1))), axis=1) #4096*101
            #print('pac_k')
            #print(s_a.shape)

            # if reward --> get rewards for the same sequence appended together 
            if flag == 'reward':
                s_a = s_a.tolist()
                stacked_sa = [i*self.PAC_k for i in s_a]
                #print('reward')
                #print(np.asarray(stacked_sa).shape)
                
                return chainer.Variable(xp.asarray(stacked_sa, dtype=xp.float32))
            else:
                #Update D
                n_sa = s_a.shape[0]
                assert n_sa % self.PAC_k == 0
                stacks = xp.split(s_a, int(n_sa/self.PAC_k), axis=0)
                stacked = [xp.concatenate(i) for i in stacks]
                #print('update D')
                #print(np.asarray(stacked).shape)
                #quit()
                return chainer.Variable(xp.asarray(stacked, dtype=xp.float32))

        return F.concat((xp.array(states, dtype=xp.float32), xp.array(actions, dtype=xp.float32).reshape((-1,1))))
    
    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_D_output', mean_or_nan(self.D_output_mean)), 
                ('average_mod_rewards', mean_or_nan(self.mod_rewards)),
                ('average_rewards', mean_or_nan(self.rewards))] + super().get_statistics()
    
    def shuffle_dataset(self, states, actions):
        shuffled_states = []
        shuffled_actions = []
        for i in range(self.n_experts):
            shuffled_s, shuffled_a = shuffle(states[i], actions[i])
            shuffled_states.append(shuffled_s)
            shuffled_actions.append(shuffled_a)
        return shuffled_states, shuffled_actions

def gailtype_constructor(rl_algo=TRPO):
    _gail_parent = GAIL.mro()[1]
    _gail_func_dict = {func: getattr(GAIL, func) for func in dir(GAIL) if callable(getattr(GAIL, func))
                       and (not func.startswith("__") or func == '__init__')
                       and (not hasattr(_gail_parent, func)
                            or not getattr(GAIL, func) == getattr(_gail_parent, func))}
    return type("GAIL" + rl_algo.__name__.upper(), (rl_algo,), _gail_func_dict)


def get_purchase_ratio(sequence):
    return np.count_nonzero(sequence)/len(sequence)


# GAILTRPO do not work because TRPO's interface is not compatible with PPO
#GAILTRPO = gailtype_constructor(rl_algo=TRPO)
