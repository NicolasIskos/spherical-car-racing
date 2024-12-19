import torch

class RaceEnv():
    def __init__(self):
        self.num_future_dir_vectors = 4
        self.traction_limit = 2
        self.dt = 0.1
        
        self.iter_counter = 0
        self.tiles_visited = set()

    def is_pos_valid(self):
        pass

    def get_tile_id(self):
        pass

    def get_reward(self):
        pass

    def step(self, a):
        pass
