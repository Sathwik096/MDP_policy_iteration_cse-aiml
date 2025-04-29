import numpy as np

grid_size = 3
n_states = grid_size * grid_size
goal_state = [2, 2]
discount_factor = 0.9

def state_to_coord(s):
    return divmod(s, grid_size)

def coord_to_state(row, col):
    return row * grid_size + col

def Action():
    return ['up', 'down', 'left', 'right']

def reward(s):
    row, col = state_to_coord(s)
    if [row, col] == goal_state:
        return 0.0
    else:
        return -1.0

# Build transition model: P[s][a] = s' (next state)
P = {}
for s in range(n_states):
    P[s] = {}
    row, col = state_to_coord(s)
    if [row, col] == goal_state:
        for a in Action():
            P[s][a] = s
        continue
    for a in Action():
        if a == 'up':
            next_row = max(row - 1, 0)
            next_col = col
        elif a == 'down':
            next_row = min(row + 1, grid_size - 1)
            next_col = col
        elif a == 'left':
            next_row = row
            next_col = max(col - 1, 0)
        elif a == 'right':
            next_row = row
            next_col = min(col + 1, grid_size - 1)
        next_state = coord_to_state(next_row, next_col)
        P[s][a] = next_state

def value_iteration(theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            row, col = state_to_coord(s)
            if [row, col] == goal_state:
                continue
            action_values = []
            for a in Action():
                next_s = P[s][a]
                action_value = reward(s) + discount_factor * V[next_s]
                action_values.append(action_value)
            max_value = max(action_values)
            delta = max(delta, abs(max_value - V[s]))
            V[s] = max_value
        if delta < theta:
            break
    return V

def policy_evaluation(policy, theta=1e-6):
    V = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            row, col = state_to_coord(s)
            if [row, col] == goal_state:
                continue
            a = Action()[policy[s]]
            next_s = P[s][a]
            v = reward(s) + discount_factor * V[next_s]
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(V):
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        row, col = state_to_coord(s)
        if [row, col] == goal_state:
            policy[s] = 0  # arbitrary action at terminal state
            continue
        action_values = []
        for a_idx, a in enumerate(Action()):
            next_s = P[s][a]
            action_value = reward(s) + discount_factor * V[next_s]
            action_values.append(action_value)
        best_action = np.argmax(action_values)
        policy[s] = best_action
    return policy

def print_policy(policy):
    for i in range(grid_size):
        row_actions = [f'"{Action()[policy[i*grid_size + j]]}"' for j in range(grid_size)]
        print(' '.join(row_actions))
    print()

def policy_iteration():
    policy = np.zeros(n_states, dtype=int)  # initialize with all 'up'
    iteration = 0
    while True:
        iteration += 1
        print(f"Policy Iteration {iteration}:")
        print_policy(policy)
        V = policy_evaluation(policy)
        new_policy = policy_improvement(V)
        if np.array_equal(new_policy, policy):
            print("Converged Policy Iteration:")
            print_policy(new_policy)
            break
        policy = new_policy

_ = value_iteration()
print("Policy Iteration Result:")
policy_iteration()
