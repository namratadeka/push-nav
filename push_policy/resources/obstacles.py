import os
import gym
import random
import numpy as np
from glob import glob
import pybullet as p


class Obstacles:
    def __init__(self, client, test=False):
        obs_files = glob(os.path.join(os.path.dirname(__file__), 'obstacle*.urdf'))
        self.np_random, _ = gym.utils.seeding.np_random()

        if test:
            for j, obs in enumerate(obs_files):
                x = 2.5
                y = 0.7*(j - 2.5)

                p.loadURDF(fileName=obs,
                           basePosition=[x, y, 0.5],
                           physicsClientId=client)
        else:
            random.shuffle(obs_files)
            for i in range(4):
                for j, obs in enumerate(obs_files):
                    r = self.np_random.uniform(0.8, 2.5)
                    theta = ((4*i + j)*20)%(2*np.pi)
                    x = r*np.cos(theta)
                    y = r*np.sin(theta)

                    p.loadURDF(fileName=obs,
                               basePosition=[x, y, 0.5],
                               physicsClientId=client)
