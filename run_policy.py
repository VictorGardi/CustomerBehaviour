import os
import gym
import custom_gym
import chainer
import itertools
import chainerrl
import numpy
import json
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from main import A3CFFSoftmax
from chainerrl.misc.batch_states import batch_states

directory = 'results/gail/discrete_events/1_expert(s)/case_2/tuesday_0225/hist64_exp256'
sample_length = 20000
n_experts = 1
n_last_days = 7

possible_val_states = [list(x) for x in itertools.product([0, 1], repeat=n_last_days+1)]

############################
##### Helper functions #####
############################

def get_val_states(states, actions, n):
    n_trajectories = len(states)

    observed_val_states = []

    for i in range(n_trajectories):
        for temp_state, temp_action in zip(states[i], actions[i]):
            # Extract the last n purchases
            last_week = temp_state[-n:]
            val_state = [int(x) for x in last_week]
            val_state.append(temp_action)
            observed_val_states.append(val_state)

    return observed_val_states

def get_counts(observed_val_states, possible_val_states):
    counts = len(possible_val_states) * [0]
    for temp in observed_val_states:
        i = possible_val_states.index(temp)
        counts[i] += 1
    return counts

def print_val_states(counts, possible_val_states, n):
    indices = np.argpartition(counts, -n)[-n:]
    indices = indices[np.argsort(np.array(counts)[indices])]
    for i in indices:
        print('Index: %d' % i)
        print('Counts: %d' % counts[i])
        print('Validation state:')
        print(np.array(possible_val_states)[i])

##############################
##### Sample from policy #####
##############################

# Load parameters and settings
args_path = join(directory, 'args.txt')
args = json.loads(open(args_path, 'r').read())

env = gym.make('discrete-buying-events-v0')
env.initialize_environment(
    case=args['state_rep'], 
    n_historical_events=args['n_historical_events'], 
    episode_length=sample_length,
    n_demos_per_expert=1,
    n_expert_time_steps=sample_length, 
    agent_seed=args['agent_seed']
    )

# Initialize model and observation normalizer
model = A3CFFSoftmax(args['n_historical_events'], 2, hidden_sizes=(args['batchsize'], args['batchsize']))
obs_normalizer = chainerrl.links.EmpiricalNormalization(args['n_historical_events'], clip_threshold=5)

# Load model and observation normalizer
model_directory = [x[0] for x in os.walk(directory)][1]
chainer.serializers.load_npz(join(model_directory, 'model.npz'), model)
chainer.serializers.load_npz(join(model_directory, 'obs_normalizer.npz'), obs_normalizer)

xp = numpy
phi = lambda x: x

agent_states = []
agent_actions = []

obs = env.reset().astype('float32')  # Initial state
agent_states.append(obs)
done = False
while not done:
    # print(obs)

    b_state = batch_states([obs], xp, phi)
    b_state = obs_normalizer(b_state, update=False)

    with chainer.using_config('train', False), chainer.no_backprop_mode():
        action_distrib, value = model(b_state)
        action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]

    # print(action)

    agent_actions.append(action)

    new_obs, _, done, _ = env.step(action)
    obs = new_obs.astype('float32')

    if not done: agent_states.append(obs)

agent_val_states = get_val_states([agent_states], [agent_actions], n_last_days)
agent_counts = get_counts(agent_val_states, possible_val_states)

##################################
##### Sample from true model #####
##################################

if os.path.exists(join(os.getcwd(), 'expert_trajectories.npz')):
    trajectories = np.load('expert_trajectories.npz')
else:
    trajectories = env.generate_expert_trajectories(
        n_experts=n_experts, 
        out_dir='.',
        seed=True, 
        eval=False
        )

expert_states = trajectories['states']
expert_actions = trajectories['actions']

expert_val_states = get_val_states(expert_states, expert_actions, n_last_days)
expert_counts = get_counts(expert_val_states, possible_val_states)

# print_val_states(expert_counts, possible_val_states, n=9)

##########################
##### Compare result #####
##########################

_, ax = plt.subplots()
ax.plot(agent_actions)
plt.show()

quit()

x = range(len(possible_val_states))

_, (ax1, ax2) = plt.subplots(1, 2)
ax1.bar(x, expert_counts)
ax2.bar(x, agent_counts)
plt.show()
