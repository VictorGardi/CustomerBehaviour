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
from custom_gym.envs.discrete_buying_events import Case21, Case22, Case23, Case24, Case31, Case221, Case222, Case7, Case4, Case71, Case17, Case81


class MMCTGAIL(PPO):
    def __init__(self, env, discriminator, demonstrations, args, discriminator_loss_stats_window=1000, **kwargs):
        # super take arguments for dynamic inheritance
        super(self.__class__, self).__init__(**kwargs)

        self.env = env
        self.noise = args.noise
        self.n_experts = args.n_experts
        self.episode_length = args.episode_length
        self.dummy_D = args.show_D_dummy
        self.adam_days = args.adam_days
        self.n_historical = args.n_historical_events
        self.iteration = 0
        self.max_iteration = args.n_experts*args.episode_length/args.update_interval #2
        self.n_update_experts = int(args.update_interval/args.episode_length) #2
        self.discriminator = discriminator
            
        self.demo_states = [*demonstrations['states']]
        self.demo_actions = [*demonstrations['actions']]

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

        dataset_states = []
        dataset_actions = []
        if self.iteration >= self.max_iteration: self.iteration = 0
        low_lim = self.iteration*self.n_update_experts #0
        high_lim = low_lim + self.n_update_experts #1

        for agent in range(self.n_update_experts): #Collect dataset for generated data
            min_idx = agent*self.episode_length
            max_idx = (agent + 1)*self.episode_length 
            dataset_states.append([dataset[i]['state'] for i in range(min_idx, max_idx)])
            dataset_actions.append([dataset[i]['action'] for i in range(min_idx, max_idx)])

        dataset_states = np.array(dataset_states)
        dataset_actions = np.array(dataset_actions)
        demo_states = []
        demo_actions = []
        for i,expert in enumerate(range(low_lim, high_lim)):
            demo_states.append(self.demo_states[expert]) # Collect the correct corresponding expert data
            demo_actions.append(self.demo_actions[expert])

        loss_mean = 0
        len_state = demo_states[0][0].size

        n_mb = int(self.episode_length*self.n_update_experts/self.minibatch_size)
        demo_states = np.array(demo_states).reshape((-1, len_state))
        dataset_states = dataset_states.reshape((-1, len_state))
        demo_actions = np.array(demo_actions).reshape((-1, 1))
        dataset_actions = dataset_actions.reshape((-1, 1))

        for epoch in range(self.epochs):
            chosen_indices = xp.random.permutation(self.episode_length*self.n_update_experts)

            for indices in np.split(chosen_indices, n_mb):
                mb_demo_states = np.take(demo_states, indices, axis=0)
                mb_demo_actions = np.take(demo_actions, indices, axis=0)
                mb_states = np.take(dataset_states, indices, axis=0)
                mb_actions = np.take(dataset_actions, indices, axis=0)

                if self.obs_normalizer:
                    mb_states = self.obs_normalizer(mb_states, update=False)
                    mb_demo_states = self.obs_normalizer(mb_demo_states, update=False)

                self.discriminator.train(self.convert_data_to_feed_discriminator(mb_demo_states, mb_demo_actions),
                                        self.convert_data_to_feed_discriminator(mb_states, mb_actions))
                loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)

                self.discriminator_loss_record.append(float(loss_mean.array))
        self.iteration += 1

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
                D_outputs = self.discriminator.get_rewards(self.convert_data_to_feed_discriminator(states, actions)).array
            
            self.D_output_mean.append(float(np.mean(D_outputs)))
            
            s_a = np.concatenate((states, actions.reshape((-1,1))), axis=1)

            mod_rewards_temp = []
            rewards_temp = []
            i = 0
            for episode in self.memory:
                for transition in episode:
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

    def convert_data_to_feed_discriminator(self, states, actions):

        xp = self.model.xp
        if self.noise > 0:
            states = states.astype(xp.float32)
            states += xp.random.normal(loc=0., scale=self.noise, size=states.shape)
            actions = actions.astype(xp.float32)
            actions += xp.random.normal(loc=0., scale=self.noise, size=actions.shape)


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

        if not self.dummy_D:
            if isinstance(self.env.case, Case17):
                pass
            elif not isinstance(self.env.case, Case21):
                states = [s[self.env.n_experts:] for s in states]
            elif isinstance(self.env.case, Case7):
                states = [s[2+self.adam_days:] for s in states]
            elif isinstance(self.env.case, Case71):
                states = [s[self.adam_days:] for s in states]

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
