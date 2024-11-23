import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from race_05_env import Race05Env

# for dbg purposes
torch.manual_seed(0)
#torch.autograd.set_detect_anomaly(True)

def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(hidden_sizes=[32], lr=0, nlogits=0,
          epochs=100000, batch_size=1000, render=False):

    env = Race05Env()

    # pos and velocity, two dimensions
    action_dim = nlogits
    q_dim = 1
    obs_dim = 4
    model = mlp(sizes=[obs_dim]+hidden_sizes+[nlogits + q_dim])
    model_optimizer = Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.HuberLoss()

    def compute_loss(critic_vals, acts, returns):
        critic_loss = loss_func(critic_vals, returns)
        actor_loss = -torch.log(acts) * (returns - critic_vals)
        return actor_loss.mean() + critic_loss

    def get_expected_returns(returns, ep_len):
        gamma = 0.99
        for i in range(2, ep_len+1):
            returns[-i] = gamma**(i-1)*returns[-i] + returns[-i+1]

    def train_one_epoch(epoch_idx):
        # obs = [x, y, vx, vy]
        batch_obs = torch.empty((batch_size, 4))
        batch_obsp = torch.empty((batch_size, 4))
        batch_acts = torch.empty(batch_size)
        batch_critic_vals = torch.empty(batch_size)
        batch_returns = torch.empty(batch_size)
        batch_lens = torch.empty(batch_size)

        obs, _, _ = env.reset(True)
        done = False
        last_ep_iteration = 0
        iteration = 0
        num_episodes = 0

        while True:
            if done or iteration >= batch_size:
                batch_lens[num_episodes] = iteration - last_ep_iteration
                num_episodes += 1

                get_expected_returns(batch_returns[last_ep_iteration:iteration], iteration - last_ep_iteration)

                # reset for next episode
                #print('done', iteration - last_ep_iteration, iteration, last_ep_iteration)
                obs, done, last_ep_iteration = env.reset()

                if iteration >= batch_size:
                    break
                else:
                    continue

            # save obs
            batch_obs[iteration,:] = obs

            # act in the environment
            res = model(obs)

            # this is DPG. We can't handle continuous action spaces yet,
            # so break up the max accel traction circle into 16 possible directions and choose one.
            # res[0:nlogits] = [theta_logits], res[nlogits:] = Q val
            act_logits, critic_val = res[0:nlogits], res[nlogits:]
            act = torch.distributions.categorical.Categorical(logits=act_logits).sample().item()
            act_prob = torch.softmax(act_logits, dim=0)[act]

            # convert theta logit to accel vector
            theta = act/nlogits * 2*np.pi
            a = torch.tensor([np.cos(theta), np.sin(theta)]) 
            
            # progress agent
            obs, rew, done = env.step(a)

            #print(res, obs, rew)
            
            batch_obsp[iteration,:] = obs
            batch_acts[iteration] = act_prob
            batch_critic_vals[iteration] = critic_val
            batch_returns[iteration] = rew
            iteration += 1
        
        model_optimizer.zero_grad()
        model_batch_loss = compute_loss(batch_critic_vals, batch_acts, batch_returns)
        model_batch_loss.backward()
        model_optimizer.step()

        print(batch_obs[-120:-1,:])

        return model_batch_loss, batch_lens.resize_(num_episodes)

    # training loop
    for i in range(epochs):
        model_batch_loss, batch_lens = train_one_epoch(i)
        print('epoch: %3d \t loss: %.3f \t ep_len: %.3f'%
                (i, model_batch_loss, torch.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlogits', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    train(lr=args.lr, nlogits=args.nlogits)