import torch
import random
import matplotlib.pyplot as plt
from plot_trajectories import plot as plot

torch.set_printoptions(threshold=10000, sci_mode=False, linewidth=100)

n_actions = 5
n_episodes = 16
n_runs = 1
ep_len_limit = 8192

# initialize state values 
n_xtiles = 80
n_ytiles = 80
n_xspeeds = 6
n_yspeeds = 6

state_vals = torch.empty((n_xtiles, n_ytiles, n_xspeeds, n_yspeeds))

# sensitive to initial values. Too high and the algo is far too optimistic 
# and never converges. Too low and the algo doesn't explore enough early on. 
# Policy is attracted to known states ven when they're suboptimal.
state_vals[:,:,:,:] = -500

# hyperparams
alpha = 1e-2
gamma = 1
eps = 0.02

init_state = torch.Tensor([10, 0, 0, 0]).to(dtype=torch.long)

run_avg_ep_lens = torch.zeros(n_episodes)
min_ep_len = ep_len_limit
min_ep_len_states = torch.empty((ep_len_limit+1, 4))

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
            q = state_vals[tuple(s)]
            q_vals[a] = q

        # if all action values are the same, choose randomly  
        if(torch.std(q_vals).isclose(torch.zeros(1))):
            action = int(random.random() * n_actions)
        else:
            action = torch.argmax(q_vals)

    return action

for run in range(n_runs):
    # main loop
    ep_lens = torch.empty(n_episodes)
    for episode in range(n_episodes):
        done = False
        step = 0
        state = init_state.clone()
        visited_states = torch.empty((ep_len_limit+1, 4), dtype=torch.long)
        visited_states[0] = state
        returns = torch.empty(ep_len_limit)
        while(not done and step < ep_len_limit):
            action = get_action(state)
            state, done = get_new_state(state, action)
            visited_states[step + 1] = state
            returns[step] = -1
            step += 1

        visited_states.resize_((step+1, 4))
        returns.resize_(step)

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
        
        # convert returns to prefix sum (potentially discounted)
        for i in range(2, step+1):
            returns[-i] = gamma**(i-1)*returns[-i] + returns[-i+1]

        # update policy by updating Q values, first visit
        state_set = set()
        for i in range(2, step-1):
            s = visited_states[-i]
            if(s not in state_set):
                state_set.add(tuple(s))
                ret = returns[-i+1]
                # For now, simply have each state val be the average of its sampled future returns
                state_vals[tuple(s)] += alpha * (ret - state_vals[tuple(s)])

    run_avg_ep_lens += ep_lens
run_avg_ep_lens /= n_runs

print(torch.min(ep_lens))
#print(min_ep_len_states[0:min_ep_len+1])
plot(last_few_episode_states, min_ep_len_states, min_ep_len, init_state)
