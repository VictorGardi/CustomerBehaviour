import chainer
from sklearn.utils import shuffle
import numpy as np
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents import PPO
from itertools import chain
import collections
import chainer.functions as F
from chainerrl.policies import SoftmaxPolicy
from custom_gym.envs.discrete_buying_events import Case22, Case23, Case24, Case31, Case221, Case222, Case7
from customer_behaviour.algorithms.irl.airl.discriminator import Discriminator
from customer_behaviour.algorithms.irl.common.utils.mean_or_nan import mean_or_nan
from customer_behaviour.algorithms.irl.common.utils import get_states_actions_next_states



class AIRL(PPO):
    def __init__(self, env, discriminator, demonstrations, noise, n_experts, episode_length, adam_days, dummy_D, discriminator_loss_stats_window=1000, **kwargs):

        super().__init__(**kwargs)

        self.env = env
        self.dummy_D = dummy_D
        self.noise = noise
        self.n_experts = n_experts
        self.episode_length = episode_length
        self.adam_days = adam_days
        self.discriminator = discriminator
        self.demo_states, self.demo_actions, self.demo_next_states = get_states_actions_next_states(demonstrations['states'], demonstrations['actions'], case=self.env.case, xp=self.xp)

        #if isinstance(self.model.pi, SoftmaxPolicy):
        #    # action space is continuous
        #    self.demo_actions = self.demo_actions.astype(dtype=self.xp.int32)

        self.discriminator_loss_record = collections.deque(maxlen=discriminator_loss_stats_window)
        self.reward_mean_record = collections.deque(maxlen=discriminator_loss_stats_window)

    def _update(self, dataset):
        # override func
        xp = self.xp

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

        if isinstance(self.env.case, Case221) or isinstance(self.env.case, Case7) or isinstance(self.env.case, Case23):
            dataset_states = []
            dataset_actions = []
            dataset_next_states = []
            for expert in range(self.n_experts):
                min_idx = expert*self.episode_length
                max_idx = (expert + 1)*self.episode_length 
                dataset_states.append([dataset[i]['state'] for i in range(min_idx, max_idx)])
                dataset_actions.append([dataset[i]['action'] for i in range(min_idx, max_idx)])
                dataset_next_states.append([dataset[i]['next_state'] for i in range(min_idx, max_idx)])

            loss_mean = 0
            n_mb = int(self.episode_length/self.minibatch_size)
            for epoch in range(self.epochs):

                for expert in range(self.n_experts):
                    demo_states = self.demo_states[expert]
                    demo_actions = self.demo_actions[expert]
                    demo_next_states = self.demo_next_states[expert]
                    states = dataset_states[expert]
                    actions = dataset_actions[expert]
                    next_states = dataset_next_states[expert]
                    states, actions, next_states = shuffle(np.array(states), np.array(actions), np.array(next_states))
                    demo_states, demo_actions, demo_next_states = shuffle(demo_states, demo_actions, demo_next_states)

                    for demo_state, state in zip(demo_states, states):
                       demo_dummy = list(map(int, list(demo_state[:self.n_experts])))
                       dummy = list(map(int, list(state[:self.n_experts])))
                       if not demo_dummy == dummy:
                           raise NameError('States are in the wrong order!')
                       else:
                           pass # the order of expert and agent is correct
                    with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                        action_log_probs = self.get_probs(np.array(states, dtype=np.float32), np.array(actions, dtype=np.int))
                        demo_action_log_probs = self.get_probs(np.array(demo_states, dtype=np.float32), np.array(demo_actions, dtype=np.int))

                    for mb in range(n_mb):
                        min_idx = mb*self.minibatch_size
                        max_idx = (mb + 1)*self.minibatch_size
                        mb_states = states[min_idx:max_idx,:]
                        mb_actions = actions[min_idx:max_idx]
                        mb_next_states = next_states[min_idx:max_idx]
                        mb_demo_states = demo_states[min_idx:max_idx,:]
                        mb_demo_actions = demo_actions[min_idx:max_idx]
                        mb_demo_next_states = demo_next_states[min_idx:max_idx]
                        mb_action_log_probs = action_log_probs[min_idx:max_idx]
                        mb_demo_action_log_probs = demo_action_log_probs[min_idx:max_idx]

                        if not self.dummy_D:
                            mb_states = [s[self.env.n_experts:] for s in mb_states]
                            mb_next_states = [s[self.env.n_experts:] for s in mb_next_states]
                            mb_demo_states = [s[self.env.n_experts:] for s in mb_demo_states]
                            mb_demo_next_states = [s[self.env.n_experts:] for s in mb_demo_next_states]

                        if self.obs_normalizer:
                            mb_states = self.obs_normalizer(mb_states, update=False)
                            mb_demo_states = self.obs_normalizer(mb_demo_states, update=False)
                            mb_next_states = self.obs_normalizer(mb_next_states, update=False)
                            mb_demo_next_states = self.obs_normalizer(mb_demo_next_states, update=False)

                        loss = self.discriminator.train(expert_states=np.array(mb_demo_states, dtype=np.float32), expert_next_states=np.array(mb_demo_next_states, dtype=np.float32),
                                                expert_action_probs=mb_demo_action_log_probs, fake_states=np.array(mb_states, dtype=np.float32),
                                                fake_next_states=np.array(mb_next_states, dtype=np.float32), fake_action_probs=mb_action_log_probs,
                                                gamma=self.gamma)
                        loss_mean += loss / (self.epochs * self.minibatch_size)

        else:

            dataset_iter = chainer.iterators.SerialIterator(dataset, self.minibatch_size, shuffle=True)
            loss_mean = 0
            while dataset_iter.epoch < self.epochs:
                # create batch for this iter
                batch = dataset_iter.__next__()
                states = self.batch_states([b['state'] for b in batch], xp, self.phi)
                next_states = self.batch_states([b['next_state'] for b in batch], xp, self.phi)
                actions = xp.array([b['action'] for b in batch])

                # create batch of expert data for this iter
                demonstrations_indexes = np.random.permutation(len(self.demo_states))[:self.minibatch_size]
                demo_states, demo_actions, demo_next_states = [d[demonstrations_indexes]
                                                            for d in (self.demo_states, self.demo_actions,
                                                                        self.demo_next_states)]

                states, demo_states, next_states, demo_next_states = [(self.obs_normalizer(d, update=False)
                                                                    if self.obs_normalizer else d)
                                                                    for d in [states, demo_states,
                                                                                next_states, demo_next_states]]
                                                                                
                with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                    action_log_probs = self.get_probs(states, actions)
                    demo_action_log_probs = self.get_probs(demo_states, demo_actions)

                loss = self.discriminator.train(expert_states=demo_states, expert_next_states=demo_next_states,
                                                expert_action_probs=demo_action_log_probs, fake_states=states,
                                                fake_next_states=next_states, fake_action_probs=action_log_probs,
                                                gamma=self.gamma)
                loss_mean += loss / (self.epochs * self.minibatch_size)
        self.discriminator_loss_record.append(float(loss_mean.array))
        super()._update(dataset)

    def _update_if_dataset_is_ready(self):
        # override func
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else sum(
                len(episode) for episode in self.batch_last_episode)))
        if dataset_size >= self.update_interval:
            self._flush_last_episode()

            # update reward in self.memory
            transitions = list(chain(*self.memory))
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                rewards = self.discriminator.get_rewards(self.xp.asarray(np.concatenate([transition['state'][None]  # [None] adds an extra [] around the states
                                                                         for transition in transitions]))).array
            self.reward_mean_record.append(float(np.mean(rewards)))

            i = 0
            for episode in self.memory:
                for transition in episode:

                    transition['reward'] = float(rewards[i])
                    i += 1
            assert self.memory[0][0]['reward'] == float(rewards[0]), 'rewards is not replaced.'

            dataset = _make_dataset(
                    episodes=self.memory,
                    model=self.model,
                    phi=self.phi,
                    batch_states=self.batch_states,
                    obs_normalizer=self.obs_normalizer,
                    gamma=self.gamma,
                    lambd=self.lambd,
                )
            assert len(dataset) == dataset_size
            self._update(dataset)
            self.memory = []

    def convert_data_to_feed_discriminator(self, states, actions, flag='loss'):

        xp = self.model.xp

        if self.noise > 0:
            states = states.astype(xp.float32)
            states += xp.random.normal(loc=0., scale=self.noise, size=states.shape)
            actions = actions.astype(xp.float32)
            actions += xp.random.normal(loc=0., scale=self.noise, size=actions.shape)
        
        if not self.dummy_D:
            states = [s[self.env.n_experts:] for s in states]

        return F.concat((xp.array(states, dtype=xp.float32), xp.array(actions, dtype=xp.float32).reshape((-1,1))))

    def get_probs(self, states, actions):
        target_distribs, _ = self.model(states)
        return target_distribs.log_prob(actions)

    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_rewards', mean_or_nan(self.reward_mean_record))] + super().get_statistics()




