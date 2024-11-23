import torch

class RaceEnv():
    def __init__(self, x, y, vx, vy):
        self.init_state = torch.tensor([x, y, vx, vy], dtype=torch.float32, requires_grad=False)
        self.state = torch.tensor([x, y, vx, vy], dtype=torch.float32, requires_grad=False)
        self.iter_counter = 0

    def reset(self, reset_iter_counter=False):
        self.state.copy_(self.init_state)
        final_iter = self.iter_counter
        if(reset_iter_counter):
            self.iter_counter = 0

        # obs, done, iteration
        return self.init_state, False, final_iter

    def is_pos_valid(self):
        pass

    def get_speed_along_track_dir(self):
        pass

    def step(self, a):
        pass
