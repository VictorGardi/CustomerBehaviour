import chainer
import numpy as np
from chainerrl.agents.ppo import _make_dataset
from chainerrl.agents import PPO
from itertools import chain
import collections
from chainerrl.policies import SoftmaxPolicy

from customer_behaviour.algorithms.irl.airl.discriminator_test import Discriminator
from customer_behaviour.algorithms.irl.common.utils.mean_or_nan import mean_or_nan
from customer_behaviour.algorithms.irl.common.utils import get_states_actions_next_states



class AIRL(PPO):
    def __init__(self, discriminator: Discriminator, demonstrations, discriminator_loss_stats_window=1000, **kwargs):

        super().__init__(**kwargs)
        self.discriminator = discriminator

        self.demo_states, self.demo_actions, self.demo_next_states = \
            get_states_actions_next_states(demonstrations['states'], demonstrations['actions'], xp=self.xp)
        if isinstance(self.model.pi, SoftmaxPolicy):
            # action space is continuous
            self.demo_actions = self.demo_actions.astype(dtype=self.xp.int32)

        self.discriminator_loss_record = collections.deque(maxlen=discriminator_loss_stats_window)
        self.reward_mean_record = collections.deque(maxlen=discriminator_loss_stats_window)

    def _update(self, dataset):
        # override func
        xp = self.xp

        if self.obs_normalizer:
            self._update_obs_normalizer(dataset)

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

            # Get random expert state-action pair
            demo_states, demo_actions, demo_next_states = [d[demonstrations_indexes]
                                                           for d in (self.demo_states, self.demo_actions,
                                                                     self.demo_next_states)]
            # Normalize if chosen
            states, demo_states, next_states, demo_next_states = [(self.obs_normalizer(d, update=False)
                                                                  if self.obs_normalizer else d)
                                                                  for d in [states, demo_states,
                                                                            next_states, demo_next_states]]

            # Get the probabilities for actions for expert and agent from policy net, i.e. self.model
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                action_log_probs = self.get_probs(states, actions).data
                demo_action_log_probs = self.get_probs(demo_states, demo_actions).data

            # Merge together expert's and agent's states, actions and probabilites and create a target array with ones and zeros
            batch_states = np.concatenate((states, demo_states))
            batch_next_states = np.concatenate((next_states, demo_next_states))
            batch_actions = np.concatenate((actions, demo_actions))
            batch_probs = np.concatenate((action_log_probs, demo_action_log_probs))
            targets = np.ones((len(actions) + len(demo_actions), 1))
            targets[:len(actions)] = 0 # the target for fake data is 0 and 1 for expert (true) data

            loss = self.discriminator.train(states = batch_states, actions = batch_actions, action_logprobs = batch_probs, next_states = batch_next_states, targets = targets, xp = self.xp)

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

            # Get agent's states and actions. Each list should be update_interval long
            saved_states = [transition['state'][None] for transition in transitions]
            saved_actions = [transition['action'][None] for transition in transitions]

            # Create state-action pairs, i.e. add a corresponding action to the state list. Each state-action pair
            # should be n_historical events + 1 long for a discrete action, i.e. buy/not buy
            state_action = []
            for state, action in zip(saved_states, saved_actions):
                action = np.array([0, 1]) if action == 0 else np.array([0, 1])
                array = np.append(state, action)
                state_action.append(array.reshape((-1,1)))

            # Get rewards for all s-a pairs
            with chainer.configuration.using_config('train', False), chainer.no_backprop_mode():
                rewards = self.discriminator.get_rewards(self.xp.asarray([s_a.T.astype('float32') for s_a in state_action])).array
                #rewards = self.discriminator.get_rewards(state_action.T.astype('float32')).data

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

    def get_probs(self, states, actions):
        target_distribs, _ = self.model(states)
        return target_distribs.log_prob(actions)

    def get_statistics(self):
        return [('average_discriminator_loss', mean_or_nan(self.discriminator_loss_record)),
                ('average_rewards', mean_or_nan(self.reward_mean_record))] + super().get_statistics()




