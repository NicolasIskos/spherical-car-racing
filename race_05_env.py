import torch
import numpy as np

from race_env import RaceEnv

class Race05Env(RaceEnv):
    def __init__(self):
        super().__init__()
        
        # initial conditions prescribed by race 05
        x, y, vx, vy = 10, 0, 0, 0

        # 100 tiles vertical up to corner, then 120 horizontal tiles, 1 finish tile
        self.num_tiles = 100 + 120 + 1

        # add self.num_future_dir_vectors to length so we don't read off 
        # the end when getting close to the end of the track
        self.dir_vectors = torch.empty((self.num_tiles + self.num_future_dir_vectors, 2))

        # direction vectors for each track tile
        for i in range(self.dir_vectors.shape[0]):
            if i < 100:
                self.dir_vectors[i:i+1,:] = torch.Tensor([0, 1]).to(dtype=torch.float32)
            else:
                self.dir_vectors[i:i+1,:] = torch.Tensor([1, 0]).to(dtype=torch.float32)

        # encode some information about where the track boundaries are
        wall_distances = torch.Tensor([10, 10, 0, 150]).to(dtype=torch.float32)

        self.init_state = torch.cat(
            (torch.tensor([x, y, vx, vy], dtype=torch.float32),
             torch.flatten(self.dir_vectors[0:self.num_future_dir_vectors]),
             wall_distances))

        self.obs_dim = self.init_state.shape[0]

        self.state = self.init_state.clone()

    def is_pos_valid(self):
        x = self.state[0]
        y = self.state[1]
        
        if 0 <= y and y <= 100 and 0 <= x and x <= 20:
            return True
        if 100 <= y and y <= 150 and 0 <= x:
            return True
        return False

    def normalize(self, state):
        # normalize distances by 120, velocity by 10

        #state[0] /= 120
        #state[1] /= 120
        #state[2] /= 120
        #state[3] /= 120
        state[4+2*self.num_future_dir_vectors] /= 120
        state[4+2*self.num_future_dir_vectors+1] /= 120
        state[4+2*self.num_future_dir_vectors+2] /= 120
        state[4+2*self.num_future_dir_vectors+3] /= 120

        return state


    def step(self, a_in):
        statep = torch.empty_like(self.state)
        tile_id = self.get_tile_id()

        # update position and velocity
        # TODO: Use a better numerical integration method like rk
        statep[2:4] = self.state[2:4] + self.traction_limit * a * self.dt
        statep[0:2] = self.state[0:2] + self.state[2:4] * self.dt

        # append future track direction vectors to state
        statep[4:4+2*self.num_future_dir_vectors] = torch.flatten(
            self.dir_vectors[tile_id+1:tile_id+1+self.num_future_dir_vectors])

        # compute track boundary information
        x, y = self.state[0], self.state[1]
        xleft, ybottom = x, y
        xright, ytop = 120-xleft, 150-ybottom
        wall_distances = torch.Tensor([xleft, xright, ybottom, ytop]).to(dtype=torch.float32)

        # append track boundary information to state
        statep[4+2*self.num_future_dir_vectors:4+2*self.num_future_dir_vectors+4] = wall_distances
        
        # update stored state
        self.state.copy_(statep)
        
        crashed = not self.is_pos_valid()
        finished = (tile_id == self.num_tiles - 1)
        done = crashed or finished
        rew = self.get_reward()
        self.iter_counter += 1

        # before returning, normalize state information
        statep = self.normalize(statep)

        return statep, rew, done, tile_id
    
    def reset(self, reset_iter_counter=False):
        self.state.copy_(self.init_state)
        final_iter = self.iter_counter
        if(reset_iter_counter):
            self.iter_counter = 0

        self.tiles_visited = set()

        # obs, done, iteration
        statep = torch.empty_like(self.state)

        return self.normalize(statep), False, final_iter

    def get_tile_id(self):
        x, y = self.state[0], self.state[1]
        
        if(y < 100):
            return int(y)
        else:
            return min(100 + int(x), 220)

    def get_reward(self):
        tile_id = self.get_tile_id()

        if(tile_id not in self.tiles_visited):
            reward = 1
        else:
            reward = 0
        
        self.tiles_visited.add(tile_id)

        return reward
 