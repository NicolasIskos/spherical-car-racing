import torch
import matplotlib.pyplot as plt

def plot(last_few_episode_states, min_ep_len_states, min_ep_len, init_state, bounds):
    n_lastfew = len(last_few_episode_states)
    for i in range(n_lastfew):
        ep_len = last_few_episode_states[i][1]
        episode_states = last_few_episode_states[i][0]
        episode_x = last_few_episode_states[i][0][0:ep_len+1,0:1]
        episode_y = last_few_episode_states[i][0][0:ep_len+1,1:2]
        
        # process episodes
        for k in range(1, ep_len):
            # restart pos
            if(torch.isclose(episode_states[k], init_state).all()):
                episode_x = episode_x[0:k]
                episode_y = episode_y[0:k]
                break

        plt.plot(episode_x, episode_y, color='gray')
    plt.plot(bounds[0], bounds[1], color='black')
    plt.plot(min_ep_len_states[0:min_ep_len+1,0:1], min_ep_len_states[0:min_ep_len+1,1:2], color='red', linestyle='--')
    plt.show()

def show_state_heatmap(state_vals):
    plt.imshow(torch.mean(state_vals, dim=(2,3)).T, origin='lower', vmin=-101, vmax=-98)
    plt.colorbar()
    plt.show()


#for i in range(min_ep_len+1):
#    print(min_ep_len_states[i], state_vals[tuple(min_ep_len_states[i])])
