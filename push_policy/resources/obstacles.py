import os
import gym
import numpy as np
from glob import glob
import pybullet as p


class Obstacles:
    def __init__(self, client):
        obs_files = glob(os.path.join(os.path.dirname(__file__), 'obstacle*.urdf'))
        self.np_random, _ = gym.utils.seeding.np_random()

        for i in range(4):
            for obs in obs_files:
                r = self.np_random.uniform(0.7, 2.5)
                theta = self.np_random.uniform(0, 2*np.pi)
                x = r*np.cos(theta)
                y = r*np.sin(theta)

                p.loadURDF(fileName=obs,
                           basePosition=[x, y, 0],
                           physicsClientId=client)
