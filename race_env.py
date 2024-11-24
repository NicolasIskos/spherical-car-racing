import torch

class RaceEnv():
    def __init__(self):
        self.num_future_dir_vectors = 4
        self.traction_limit = 2
        self.dt = 0.1
        
        self.iter_counter = 0
        self.tiles_visited = set()

    def reset(self, reset_iter_counter=False):
        self.state.copy_(self.init_state)
        final_iter = self.iter_counter
        if(reset_iter_counter):
            self.iter_counter = 0

        self.tiles_visited = set()

        # obs, done, iteration
        return self.init_state, False, final_iter

    def is_pos_valid(self):
        pass

    def get_tile_id(self):
        pass

    def get_reward(self):
        pass

    def step(self, a):
        pass
