import torch
import math
import random
import matplotlib.pyplot as plt
from plot_trajectories import plot as plot
from plot_trajectories import show_state_heatmap as show_state_heatmap

torch.set_printoptions(threshold=10000, sci_mode=False, linewidth=100)

n_actions = 5
n_episodes = 8192
n_runs = 1
ep_len_limit = 8192

# initialize state values 
n_xtiles = 80
n_ytiles = 80
n_xspeeds = 6
n_yspeeds = 6
bounds = ([20, 20, 80], [0, 60, 60])

# TD(0) semi-gradient weights
n_features = 64
weights = torch.normal(torch.zeros(n_features), torch.ones(n_features))

# hyperparams
alpha = 1e-2
gamma = 1
eps=0.5

init_state = torch.Tensor([10, 0, 0, 0]).to(dtype=torch.float)

run_avg_ep_lens = torch.zeros(n_episodes)
min_ep_len = ep_len_limit
min_ep_len_states = torch.empty((ep_len_limit+1, 4), dtype=torch.float)

n_lastfew = 2
last_few_episode_states = []

def get_new_state(state, action):
    # if action causes us to finish the race, the episode is finished
    statep = state.clone()

    # update velocity, don't let velocity fall to zero
    if(action == 0 and statep[2] < n_xspeeds - 1):
        statep[2] += 1
    elif(action == 1 and statep[2] > 0 and statep[3] > 0):
        statep[2] -= 1
    elif(action == 2 and statep[3] < n_yspeeds - 1):
        statep[3] += 1
    elif(action == 3 and statep[3] > 0) and statep[2] > 0:
        statep[3] -= 1
    
    # update position
    statep[0:2] += statep[2:4]
    
    # if action causes a crash, set us back to the start and continue episode
    x, y = statep[0], statep[1]
    if(x < 0 or y >= 80 or (x >= 20 and y < 60)):
        statep.copy_(init_state)
        return statep, False

    # if we finish, pile ourselves.
    if(x >= n_xtiles-1):
        # x pos cannot exceed finish line
        statep[0] = n_xtiles - 1
        return statep, True

    return statep, False

def get_features(state):
    """
    state is [x, y, vx, vy]
    Need to model a fairly complex state-value surface. Spam with lots of polynomial combinations.
    """
    x, y, vx, vy = state
    
    f = torch.empty(int(math.sqrt(n_features)), dtype=torch.float)
    f[0] = x            # distance from left boundary
    f[1] = 20 - x       # distance from right boundary before corner
    f[2] = n_xtiles - x # distance from right boundary after corner
    f[3] = y            # distance from bottom boundary
    f[4] = y - 60       # distance from bottom boundary after corner
    f[5] = n_ytiles - y # distance from top boundary
    f[6] = vx           # x speed
    f[7] = vy           # y speed

    # cross over all features with each other
    f_crossed = 1e-3*torch.outer(f, f).flatten()

    return f_crossed

def estimate_state_value(state):
    f = get_features(state)
    v = torch.dot(weights, f)
    return v

def get_action(state):
    # eps greedy. With probability 1-eps, choose the maximal action.
    # With probability eps, choose a random action, to ensure exploration.
    x = random.random()
    if(x < eps):
        action = int(random.random() * n_actions)
    else:
        q_vals = torch.empty(n_actions)
        for a in range(n_actions):
            s, _ = get_new_state(state, a)
            q = estimate_state_value(s)
            q_vals[a] = q

        print('qvals=',q_vals)
        # if all action values are the same, choose randomly  
        if(torch.std(q_vals).isclose(torch.zeros(1))):
            action = int(random.random() * n_actions)
        else:
            action = torch.argmax(q_vals)

    return action

def apply_update(old_state, state, done, ret):
    # TODO: fix this awful code
    global weights
    # we are updating the old_state
    old_est = estimate_state_value(old_state)
    est = estimate_state_value(state)
    grad = get_features(old_state)
    #weights += alpha * (ret + gamma*est - old_est) * grad
    # L2 norm
    weights += alpha * ((ret + gamma*est - old_est) * grad - 0.1*2*weights)
    #print('weights=',weights)

def evaluate_policy():
    # turn off e-greedy for policy evaluation
    eps = 0

    done = False
    step = 0
    state = init_state.clone()
    visited_states = torch.empty((ep_len_limit+1, 4), dtype=torch.long)
    visited_states[0] = state
    while(not done and step < ep_len_limit):
        action = get_action(state)
        state, done = get_new_state(state, action)
        visited_states[step + 1] = state
        step += 1

    return visited_states, step

ep_lens = torch.empty(n_episodes)
for episode in range(n_episodes):
    done = False
    step = 0
    state = init_state.clone()
    visited_states = torch.empty((ep_len_limit+1, 4), dtype=torch.long)
    visited_states[0] = state

    # run episode
    while(not done and step < ep_len_limit):
        old_state = state.clone()
        action = get_action(state)
        state, done = get_new_state(state, action)
        visited_states[step + 1] = state
        step += 1

        print(state, action)

        # inline update
        apply_update(old_state, state, done, ret=-1)

    #eps *= 0.999

    visited_states.resize_((step+1, 4))

    print('finished', step)

    # save ep len
    ep_lens[episode] = step

    # keep track of best episode
    if(step < min_ep_len):
        min_ep_len = step
        min_ep_len_states[0:step+1] = visited_states

    # if last few episodes, add to log
    if n_episodes - episode <= n_lastfew:
        last_few_episode_states.append((visited_states, step))

# evaluate final policy
final_policy, final_policy_ep_len = evaluate_policy()
print(final_policy_ep_len)
print(torch.min(ep_lens))

plot(last_few_episode_states, min_ep_len_states, min_ep_len, init_state, bounds)
# Don't include finish states, because their values are very high, messes up contrast.
show_state_heatmap(state_vals[:-1,:,:,:])