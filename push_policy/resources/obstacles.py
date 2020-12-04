import os
import gym
from glob import glob
import pybullet as p


class Obstacles:
    def __init__(self, client):
        obs_files = glob(os.path.join(os.path.dirname(__file__), 'obstacle*.urdf'))
        self.np_random, _ = gym.utils.seeding.np_random()

        for i in range(4):
            for obs in obs_files:
                x = (self.np_random.uniform(-2.5, -0.5) if self.np_random.randint(2) else
                    self.np_random.uniform(0.5, 2.5))
                y = (self.np_random.uniform(-2.5, -0.5) if self.np_random.randint(2) else
                    self.np_random.uniform(0.5, 2.5))

                p.loadURDF(fileName=obs,
                           basePosition=[x, y, 0],
                           physicsClientId=client)
