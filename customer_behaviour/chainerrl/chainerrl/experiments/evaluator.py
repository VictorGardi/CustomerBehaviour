import logging
import multiprocessing as mp
import os
import statistics
import time
import os.path
import custom_gym

import numpy as np

import chainerrl

from customer_behaviour.tools.time_series_analysis import FeatureExtraction
from customer_behaviour.tools.cluster import Cluster
from customer_behaviour.tools.validation_states import get_features_from_counts
from custom_gym.envs.discrete_buying_events import Case21
from scipy.stats import entropy
# from customer_behaviour.custom_gym.custom_gym.envs.discrete_buying_events import Case21


"""Columns that describe information about an experiment.

steps: number of time steps taken (= number of actions taken)
episodes: number of episodes finished
elapsed: time elapsed so far (seconds)
mean: mean of returns of evaluation runs
median: median of returns of evaluation runs
stdev: stdev of returns of evaluation runs
max: maximum value of returns of evaluation runs
min: minimum value of returns of evaluation runs
"""

_basic_columns = ('steps', 'episodes')

# _basic_columns = ('steps', 'episodes', 'elapsed', 'mean', 'median', 'stdev', 'max', 'min')


def run_evaluation_episodes(env, agent, n_steps, n_episodes, outdir,
                            max_episode_len=None, logger=None):
    """Run multiple evaluation episodes and return returns.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    scores = []
    terminate = False
    timestep = 0

    # Get expert features
    file = os.path.abspath(os.path.join(outdir, os.pardir)) + '/eval_expert_trajectories.npz'
    data = np.load(file, allow_pickle=True)
    assert sorted(data.files) == sorted(['states', 'actions'])
    
    expert_states = data['states']
    expert_actions = data['actions']
    # expert_features = []
    # for states, actions in zip(expert_states, expert_actions):
    #     if isinstance(env.env, custom_gym.envs.DiscreteBuyingEvents):
    #         temp = FeatureExtraction(np.array(actions), case='discrete_events').get_features()
    #         # assert isinstance(env.env.case, Case21), 'Must use case 2.1 for validation measure to work'
    #         # expert_purchase, expert_no_purchase = get_features_from_counts([states], [actions])
    #         # temp.extend(expert_no_purchase)  # start by adding counts given no purchase
    #     else:
    #         raise NotImplementedError
    #     expert_features.append(temp)

    # agent_features = []

    print('----- ----- ----- ----- ----- ----- ----- ----- ----- -----')

    # entropy_purchase = []
    # entropy_no_purchase = []

    agent_states = []
    agent_actions = []

    reset = True
    i_expert = 0
    while not terminate:
        if reset:
            temp_actions = []
            temp_states = []
            _ = env.reset()

            # Initialize agent with history from ith expert
            initial_state = expert_states[i_expert][0]
            env.env.state = initial_state.copy()
            obs = np.array(env.env.state).astype('float32')

            done = False
            test_r = 0
            episode_len = 0
            info = {}
        a = agent.act(obs)
        temp_states.append(obs)
        temp_actions.append(a)
        obs, r, done, info = env.step(a)

        test_r += r
        episode_len += 1
        timestep += 1
        reset = (done or episode_len == max_episode_len
                 or info.get('needs_reset', False))
        if reset:
            logger.info('evaluation episode %s length:%s R:%s',
                        len(scores), episode_len, test_r)
            
            if i_expert < len(expert_states) - 1:
                i_expert += 1

            # Extract features from the time-series (i.e. "actions") and
            # compare this feature vector against the cluster of expert features

            # if isinstance(env.env, custom_gym.envs.DiscreteBuyingEvents):
            #     temp_features = FeatureExtraction(np.array(temp_actions), case='discrete_events').get_features()
            #     # assert isinstance(env.env.case, Case21), 'Must use case 2.1 for validation measure to work'
            #     # purchase, no_purchase = get_features_from_counts([temp_states], [temp_actions])
            #     # temp_features.extend(no_purchase)  # start by adding counts given no purchase
            # else:
            #     raise NotImplementedError

            agent_states.append(temp_states)
            agent_actions.append(temp_actions)

            # e1 = entropy(purchase, qk=expert_purchase)  # comparing with the "last expert"
            # entropy_purchase.append(e1)

            # e2 = entropy(no_purchase, qk=expert_no_purchase)
            # entropy_no_purchase.append(e2)

            # agent_features.append(temp_features)

            # cluster = Cluster(agent_features[-1], expert_features)

            # mean_dist, min_dist, max_dist = cluster.get_dist_between_clusters()
            # logger.info('mean_dist: %.1f, min_dist: %.1f, max_dist: %.1f' % (mean_dist, min_dist, max_dist))

            # print('----- ----- ----- ----- ----- ----- ----- ----- ----- -----')

            # As mixing float and numpy float causes errors in statistics
            # functions, here every score is cast to float.
            scores.append(float(test_r))
        if n_steps is None:
            terminate = len(scores) >= n_episodes
        else:
            terminate = timestep >= n_steps
        if reset or terminate:
            agent.stop_episode()

    # Compare entire clustersc
    # cluster = Cluster(agent_features, expert_features)
    # mean_dist, min_dist, max_dist = cluster.get_dist_between_clusters()
    # agent_within_SS = cluster.agent_within_SS
    # expert_within_SS = cluster.expert_within_SS

    print('----- ----- ----- ----- ----- ----- ----- ----- ----- -----')
    # print('Comparing entire clusters')
    # logger.info('mean_dist: %.1f, min_dist: %.1f, max_dist: %.1f' % (mean_dist, min_dist, max_dist))
    # print('----- ----- ----- ----- ----- ----- ----- ----- ----- -----')

    # Print cluster comparison to file
    # with open(os.path.join(outdir, 'cluster.txt'), 'a+') as f:
    #     values = [mean_dist, min_dist, max_dist, agent_within_SS, expert_within_SS]
    #     print('\t'.join(str(x) for x in values), file=f)

    # Save agent states and actions for later validation of the training process
    states_actions_file = 'states_actions_' + str(agent.optimizer.t) + '.npz'
    np.savez(os.path.join(outdir, 'states_actions', states_actions_file), states=np.array(agent_states, dtype=object),
             actions=np.array(agent_actions, dtype=object))

    # Print KL divergences
    # with open(os.path.join(outdir, 'kl_div_purchase.txt'), 'a+') as f:
    #    print('\t'.join(str(x) for x in entropy_purchase), file=f)

    # with open(os.path.join(outdir, 'kl_div_no_purchase.txt'), 'a+') as f:
    #    print('\t'.join(str(x) for x in entropy_no_purchase), file=f)

    # If all steps were used for a single unfinished episode
    if len(scores) == 0:
        scores.append(float(test_r))
        logger.info('evaluation episode %s length:%s R:%s',
                    len(scores), episode_len, test_r)
    return scores


def batch_run_evaluation_episodes(
    env,
    agent,
    n_steps,
    n_episodes,
    max_episode_len=None,
    logger=None,
):
    """Run multiple evaluation episodes and return returns in a batch manner.

    Args:
        env (VectorEnv): Environment used for evaluation.
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of total timesteps to evaluate the agent.
        n_episodes (int): Number of evaluation runs.
        max_episode_len (int or None): If specified, episodes
            longer than this value will be truncated.
        logger (Logger or None): If specified, the given Logger
            object will be used for logging results. If not
            specified, the default logger of this module will
            be used.

    Returns:
        List of returns of evaluation runs.
    """
    assert (n_steps is None) != (n_episodes is None)

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_returns = dict()
    episode_lengths = dict()
    episode_indices = np.zeros(num_envs, dtype='i')
    episode_idx = 0
    for i in range(num_envs):
        episode_indices[i] = episode_idx
        episode_idx += 1
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_len = np.zeros(num_envs, dtype='i')

    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    termination_conditions = False
    timestep = 0
    while True:
        # a_t
        actions = agent.batch_act(obss)
        timestep += 1
        # o_{t+1}, r_{t+1}
        obss, rs, dones, infos = env.step(actions)
        episode_r += rs
        episode_len += 1
        # Compute mask for done and reset
        if max_episode_len is None:
            resets = np.zeros(num_envs, dtype=bool)
        else:
            resets = (episode_len == max_episode_len)
        resets = np.logical_or(
            resets, [info.get('needs_reset', False) for info in infos])

        # Make mask. 0 if done/reset, 1 if pass
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        for index in range(len(end)):
            if end[index]:
                episode_returns[episode_indices[index]] = episode_r[index]
                episode_lengths[episode_indices[index]] = episode_len[index]
                # Give the new episode an a new episode index
                episode_indices[index] = episode_idx
                episode_idx += 1

        episode_r[end] = 0
        episode_len[end] = 0

        # find first unfinished episode
        first_unfinished_episode = 0
        while first_unfinished_episode in episode_returns:
            first_unfinished_episode += 1

        # Check for termination conditions
        eval_episode_returns = []
        eval_episode_lens = []
        if n_steps is not None:
            total_time = 0
            for index in range(first_unfinished_episode):
                total_time += episode_lengths[index]
                # If you will run over allocated steps, quit
                if total_time > n_steps:
                    break
                else:
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])
            termination_conditions = total_time >= n_steps
            if not termination_conditions:
                unfinished_index = np.where(
                    episode_indices == first_unfinished_episode)[0]
                if total_time + episode_len[unfinished_index] >= n_steps:
                    termination_conditions = True
                    if first_unfinished_episode == 0:
                        eval_episode_returns.append(
                            episode_r[unfinished_index])
                        eval_episode_lens.append(
                            episode_len[unfinished_index])

        else:
            termination_conditions = first_unfinished_episode >= n_episodes
            if termination_conditions:
                # Get the first n completed episodes
                for index in range(n_episodes):
                    eval_episode_returns.append(episode_returns[index])
                    eval_episode_lens.append(episode_lengths[index])

        if termination_conditions:
            # If this is the last step, make sure the agent observes reset=True
            resets.fill(True)

        # Agent observes the consequences.
        agent.batch_observe(obss, rs, dones, resets)

        if termination_conditions:
            break
        else:
            obss = env.reset(not_end)

    for i, (epi_len, epi_ret) in enumerate(
            zip(eval_episode_lens, eval_episode_returns)):
        logger.info('evaluation episode %s length: %s R: %s',
                    i, epi_len, epi_ret)
    return [float(r) for r in eval_episode_returns]


def eval_performance(env, agent, n_steps, n_episodes, outdir, max_episode_len=None,
                     logger=None):
    """Run multiple evaluation episodes and return statistics.

    Args:
        env (Environment): Environment used for evaluation
        agent (Agent): Agent to evaluate.
        n_steps (int): Number of timesteps to evaluate for.
        n_episodes (int): Number of evaluation episodes.
        max_episode_len (int or None): If specified, episodes longer than this
            value will be truncated.
        logger (Logger or None): If specified, the given Logger object will be
            used for logging results. If not specified, the default logger of
            this module will be used.
    Returns:
        Dict of statistics.
    """

    assert (n_steps is None) != (n_episodes is None)

    if isinstance(env, chainerrl.env.VectorEnv):
        scores = batch_run_evaluation_episodes(
            env, agent, n_steps, n_episodes,
            max_episode_len=max_episode_len,
            logger=logger)
    else:
        scores = run_evaluation_episodes(
            env, agent, n_steps, n_episodes, outdir,
            max_episode_len=max_episode_len,
            logger=logger)
    stats = dict(
        episodes=len(scores),
        mean=statistics.mean(scores),
        median=statistics.median(scores),
        stdev=statistics.stdev(scores) if len(scores) >= 2 else 0.0,
        max=np.max(scores),
        min=np.min(scores))
    return stats


def record_stats(outdir, values):
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        print('\t'.join(str(x) for x in values), file=f)


def save_agent(agent, t, outdir, logger, suffix=''):
    dirname = os.path.join(outdir, '{}{}'.format(t, suffix))
    agent.save(dirname)
    logger.info('Saved the agent to %s', dirname)


class Evaluator(object):
    """Object that is responsible for evaluating a given agent.

    Args:
        agent (Agent): Agent to evaluate.
        env (Env): Env to evaluate the agent on.
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean of returns in evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 agent,
                 env,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        assert (n_steps is None) != (n_episodes is None), \
            ("One of n_steps or n_episodes must be None. " +
             "Either we evaluate for a specified number " +
             "of episodes or for a specified number of timesteps.")
        self.agent = agent
        self.env = env
        self.max_score = np.finfo(np.float32).min
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.prev_eval_t = (self.step_offset -
                            self.step_offset % self.eval_interval)
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Write a header line first
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in self.agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

        # Create a file for saving cluster data
        with open(os.path.join(self.outdir, 'cluster.txt'), 'w') as f:
            # custom_columns = tuple(t[0] for t in self.agent.get_statistics())    # FIXA!!!
            custom_columns = ('mean_dist', 'min_dist', 'max_dist', 'agent_within_SS', 'expert_within_SS')
            # column_names = _basic_columns + custom_columns
            print('\t'.join(custom_columns), file=f)

        with open(os.path.join(self.outdir, 'kl_div_purchase.txt'), 'w') as f:
            pass

        with open(os.path.join(self.outdir, 'kl_div_no_purchase.txt'), 'w') as f:
            pass

        # Create folder where states and actions are to be saved
        os.makedirs(os.path.join(self.outdir, 'states_actions'), exist_ok=True)

    def evaluate_and_update_max_score(self, t, episodes):
        eval_stats = eval_performance(
            self.env, self.agent, self.n_steps, self.n_episodes, self.outdir,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in self.agent.get_statistics())
        mean = eval_stats['mean']
        '''
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        '''
        values = (t, episodes) + custom_values
        record_stats(self.outdir, values)
        if mean > self.max_score:
            self.logger.info('The best score is updated %s -> %s',
                             self.max_score, mean)
            self.max_score = mean
            if self.save_best_so_far_agent:
                save_agent(self.agent, "best", self.outdir, self.logger)
        return mean

    def evaluate_if_necessary(self, t, episodes):
        if t >= self.prev_eval_t + self.eval_interval:
            score = self.evaluate_and_update_max_score(t, episodes)
            self.prev_eval_t = t - t % self.eval_interval
            return score
        return None


class AsyncEvaluator(object):
    """Object that is responsible for evaluating asynchronous multiple agents.

    Args:
        n_steps (int): Number of timesteps used in each evaluation.
        n_episodes (int): Number of episodes used in each evaluation.
        eval_interval (int): Interval of evaluations in steps.
        outdir (str): Path to a directory to save things.
        max_episode_len (int): Maximum length of episodes used in evaluations.
        step_offset (int): Offset of steps used to schedule evaluations.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
    """

    def __init__(self,
                 n_steps,
                 n_episodes,
                 eval_interval,
                 outdir,
                 max_episode_len=None,
                 step_offset=0,
                 save_best_so_far_agent=True,
                 logger=None,
                 ):
        assert (n_steps is None) != (n_episodes is None), \
            ("One of n_steps or n_episodes must be None. " +
             "Either we evaluate for a specified number " +
             "of episodes or for a specified number of timesteps.")
        self.start_time = time.time()
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.outdir = outdir
        self.max_episode_len = max_episode_len
        self.step_offset = step_offset
        self.save_best_so_far_agent = save_best_so_far_agent
        self.logger = logger or logging.getLogger(__name__)

        # Values below are shared among processes
        self.prev_eval_t = mp.Value(
            'l', self.step_offset - self.step_offset % self.eval_interval)
        self._max_score = mp.Value('f', np.finfo(np.float32).min)
        self.wrote_header = mp.Value('b', False)

        # Create scores.txt
        with open(os.path.join(self.outdir, 'scores.txt'), 'a'):
            pass

    @property
    def max_score(self):
        with self._max_score.get_lock():
            v = self._max_score.value
        return v

    def evaluate_and_update_max_score(self, t, episodes, env, agent):
        eval_stats = eval_performance(
            env, agent, self.n_steps, self.n_episodes,
            max_episode_len=self.max_episode_len,
            logger=self.logger)
        elapsed = time.time() - self.start_time
        custom_values = tuple(tup[1] for tup in agent.get_statistics())
        mean = eval_stats['mean']
        values = (t,
                  episodes,
                  elapsed,
                  mean,
                  eval_stats['median'],
                  eval_stats['stdev'],
                  eval_stats['max'],
                  eval_stats['min']) + custom_values
        record_stats(self.outdir, values)
        with self._max_score.get_lock():
            if mean > self._max_score.value:
                self.logger.info('The best score is updated %s -> %s',
                                 self._max_score.value, mean)
                self._max_score.value = mean
                if self.save_best_so_far_agent:
                    save_agent(agent, "best", self.outdir, self.logger)
        return mean

    def write_header(self, agent):
        with open(os.path.join(self.outdir, 'scores.txt'), 'w') as f:
            custom_columns = tuple(t[0] for t in agent.get_statistics())
            column_names = _basic_columns + custom_columns
            print('\t'.join(column_names), file=f)

    def evaluate_if_necessary(self, t, episodes, env, agent):
        necessary = False
        with self.prev_eval_t.get_lock():
            if t >= self.prev_eval_t.value + self.eval_interval:
                necessary = True
                self.prev_eval_t.value += self.eval_interval
        if necessary:
            with self.wrote_header.get_lock():
                if not self.wrote_header.value:
                    self.write_header(agent)
                    self.wrote_header.value = True
            return self.evaluate_and_update_max_score(t, episodes, env, agent)
        return None
