import torch

from race_env import RaceEnv

class Race05Env(RaceEnv):
    def __init__(self):
        # initial conditions prescribed by race 05
        super().__init__(10, 0, 0, 20)
        self.traction_limit = 10
        self.dt = 0.1

    def is_pos_valid(self):
        x = self.state[0]
        y = self.state[1]
        
        if 0 <= y and y <= 100 and 0 <= x and x <= 20:
            return True
        if 100 <= y and y <= 120 and 0 <= x:
            return True
        return False

    def step(self, a):
        # TODO: Use a better numerical integration method like rk
        statep = torch.empty_like(self.state)
        statep[2:] = self.state[2:] + self.traction_limit * a * self.dt
        statep[0:2] = self.state[0:2] + self.state[2:] * self.dt
        self.state = statep

        done = not self.is_pos_valid()
        rew = self.get_speed_along_track_dir()        
        self.iter_counter += 1

        return statep, rew, done

    def get_speed_along_track_dir(self):
        y = self.state[1]
        v = self.state[2:]
        # Pretty dumb, but keep it simple for now
        if 0 <= y and y <= 110:
            return v[1]
        else:
            return v[0]
 