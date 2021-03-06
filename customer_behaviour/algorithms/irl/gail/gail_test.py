import chainer
import numpy as np
import collections
import chainer.functions as F
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents import PPO, TRPO
from chainerrl.policies import SoftmaxPolicy
from itertools import chain
from customer_behaviour.algorithms.irl.common.utils.mean_or_nan import mean_or_nan

class GAIL2(PPO):
    def __init__(self, discriminator, demonstrations, discriminator_loss_stats_window=None, **kwargs):
        # super take arguments for dynamic inheritance
        super(self.__class__, self).__init__(**kwargs)

        self.discriminator = discriminator

        self.demo_states = self.xp.asarray(np.asarray(list(chain(*demonstrations['states']))).astype(np.float32))
        self.demo_actions = self.xp.asarray(np.asarray(list(chain(*demonstrations['actions']))).astype(np.float32))

        self.discriminator_loss_record = collections.deque(maxlen=discriminator_loss_stats_window)
        self.reward_mean_record = collections.deque(maxlen=discriminator_loss_stats_window)

    def _update(self, dataset):
        # override func
        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)
        
        xp = self.xp

        # assert 'state' in dataset[0]
        # assert 'v_teacher' in dataset[0]

        dataset_iter = chainer.iterators.SerialIterator(
            dataset, self.minibatch_size, shuffle=True)

        loss_mean = 0

        while dataset_iter.epoch < self.epochs:
            batch = dataset_iter.__next__()
            states = self.batch_states([b['state'] for b in batch], xp, self.phi)
            #if self.obs_normalizer:
            #   states = self.obs_normalizer(states, update=False)
            actions = xp.array([b['action'] for b in batch])

            demonstrations_indexes = np.random.permutation(len(self.demo_states))[:len(states)]

            demo_states, demo_actions = [d[demonstrations_indexes] for d in (self.demo_states, self.demo_actions)]

            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
                demo_states = self.obs_normalizer(demo_states, update=False)

            # Train discriminator
            self.discriminator.train(self.convert_data_to_feed_discriminator(demo_states, demo_actions), 
                self.convert_data_to_feed_discriminator(states, actions),
                xp
                )
            loss_mean += self.discriminator.loss / (self.epochs * self.minibatch_size)
            self.discriminator_loss_record.append(float(loss_mean.array))  # .array needed to go from "variable" to raw output

        super(self.__class__, self)._update(dataset)

    def _update_if_dataset_is_ready(self):
        # override func
        dataset_size = (
            sum(len(episode) for episode in self.memory)
            + len(self.last_episode)
            + (0 if self.batch_last_episode is None else
               sum(len(episode) for episode in self.batch_last_episode)))

        if dataset_size >= self.update_interval:
            self._flush_last_episode()

            # update reward in self.memory
            transitions = list(chain.from_iterable(self.memory))
            states = self.xp.asarray(np.concatenate([transition['state'][None] for transition in transitions])) # why None?
            actions = self.xp.asarray(np.concatenate([transition['action'][None] for transition in transitions]))

            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                rewards = self.discriminator.get_rewards(self.convert_data_to_feed_discriminator(states, actions)).array

            self.reward_mean_record.append(float(np.mean(rewards)))
            i = 0
            for episode in self.memory:
                for transition in episode:
                    transition['reward'] = float(rewards[i])
                    i += 1

            if self.recurrent:
                raise NotImplementedError
            else:
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

    def convert_data_to_feed_discriminator(self, states, actions, noise_scale=0.1):
        xp = self.model.xp
        if isinstance(self.model.pi, SoftmaxPolicy):
            # if discrete action
            # self.model.pi.model.out_size = 2 now since we only have two actions: "purchase" vs "no purchase"
            actions = xp.eye(self.model.pi.model.out_size, dtype=xp.float32)[actions.astype(xp.int32)]
        if noise_scale:
            actions += xp.random.normal(loc=0., scale=noise_scale, size=actions.shape)

        # return matrix where each row is state followed by action (action is for example [1, 0] or [0, 1] but with some noise)

        return F.concat((xp.array(states), xp.array(actions)))

    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_rewards', mean_or_nan(self.reward_mean_record))] + super().get_statistics()











