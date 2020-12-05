import os
import gym
import random
import numpy as np
from glob import glob
import pybullet as p


class Obstacles:
    def __init__(self, client):
        obs_files = glob(os.path.join(os.path.dirname(__file__), 'obstacle*.urdf'))
        random.shuffle(obs_files)
        self.np_random, _ = gym.utils.seeding.np_random()

        for i in range(4):
            for j, obs in enumerate(obs_files):
                r = self.np_random.uniform(0.8, 2.5)
                theta = ((4*i + j)*20)%(2*np.pi)
                x = r*np.cos(theta)
                y = r*np.sin(theta)

                p.loadURDF(fileName=obs,
                           basePosition=[x, y, 1],
                           physicsClientId=client)
