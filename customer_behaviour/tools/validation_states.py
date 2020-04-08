import itertools
import numpy as np


def get_counts(observed_val_states, possible_val_states, normalize=False):
    counts = len(possible_val_states) * [0]
    for temp in observed_val_states:
        i = possible_val_states.index(temp)
        counts[i] += 1
    if normalize:
        if np.sum(counts) > 0: counts = list(np.array(counts) / np.sum(counts))
    return counts

def get_cond_val_states(states, actions, n):
    n_trajectories = len(states)

    purchase = []
    no_purchase = []

    for i in range(n_trajectories):
        for temp_state, temp_action in zip(states[i], actions[i]):
            # Extract the last n days
            last_n_days = temp_state[-n:]
            val_state = [int(x) for x in last_n_days]
            # val_state.append(temp_action)
            if temp_action == 1:
                purchase.append(val_state)
            else:
                no_purchase.append(val_state)

    return purchase, no_purchase

def reduce_dimensionality(val_states, max_n_purchases, keep_only_unique=False):
    '''
    val_states: [[v_{0}], [v_{1}], ..., [v_{n-2}], [v_{n-1}]]
    max_n_purchases: The maximum number of purchases that is allowed in a validation state
    '''
    indices = np.argwhere(np.sum(val_states, axis=1) > max_n_purchases)  # list of lists
    indices = [x[0] for x in indices]

    assert len(val_states) > 0
    n = len(val_states[0])
    substitute = n * [1]

    for i in indices:
        val_states[i] = substitute

    if keep_only_unique:
        temp = set(tuple(x) for x in val_states)
        return [list(x) for x in temp]
    else:
        return val_states

def sort_possible_val_states(possible_val_states):
    temp = possible_val_states.copy()
    # temp.sort(key=lambda x: sum(x))
    # temp = np.array(temp)
    temp_splitted = []
    s = 0
    while sum(len(x) for x in temp_splitted) < len(temp):
        indices = np.argwhere(np.sum(temp, axis=1) == s)
        t = [temp[i[0]] for i in indices]
        t.sort(reverse=True)
        temp_splitted.append(t)
        s += 1
    return [item for sublist in temp_splitted for item in sublist]

def get_features_from_counts(states, actions, n_last_days=7, max_n_purchases_per_n_last_days=2):
    # Get conditional validation states
    purchase, no_purchase = get_cond_val_states(states, actions, n_last_days)

    # Reduce the dimensionality by treating all validation states with more than x purchases as one single state
    try:
        purchase = reduce_dimensionality(purchase, max_n_purchases_per_n_last_days)
    except np.AxisError:
        purchase = []

    try:
        no_purchase = reduce_dimensionality(no_purchase, max_n_purchases_per_n_last_days)
    except np.AxisError:
        no_purchase = []
    
    # Get possible validation states
    possible_val_states = [list(x) for x in itertools.product([0, 1], repeat=n_last_days)]
    possible_val_states = reduce_dimensionality(possible_val_states, max_n_purchases_per_n_last_days, keep_only_unique=True)
    possible_val_states = sort_possible_val_states(possible_val_states)

    counts_purchase = get_counts(purchase, possible_val_states, normalize=False)
    counts_no_purchase = get_counts(no_purchase, possible_val_states, normalize=False)

    return counts_purchase, counts_no_purchase
