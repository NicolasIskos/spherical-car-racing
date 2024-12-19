import torch
import matplotlib.pyplot as plt

def plot(last_few_episode_states, min_ep_len_states, min_ep_len, init_state):
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
    plt.plot([20, 20, 80], [0, 60, 60], color='black')
    plt.plot(min_ep_len_states[0:min_ep_len+1,0:1], min_ep_len_states[0:min_ep_len+1,1:2], color='red', linestyle='--')
    plt.show()