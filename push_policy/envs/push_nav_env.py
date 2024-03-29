import gym
import math
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from push_policy.resources.car import Car
from push_policy.resources.plane import Plane
from push_policy.resources.goal import Goal
from push_policy.resources.obstacles import Obstacles


class PushNavEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode:str = 'GUI'):
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -.6], dtype=np.float32),
            high=np.array([1, .6], dtype=np.float32))

        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(eval("p.{}".format(mode.upper())))
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)

        self.car = None
        self.goal = None
        self.obstacle = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None

    def dist_reward(self, car_ob):
        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  (car_ob[1] - self.goal[1]) ** 2))
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0) / 10.0
        self.prev_dist_to_goal = dist_to_goal

        return reward, dist_to_goal

    def visibility_reward(self):
        h, w = self.car.segmask.shape[:2]
        viz_pixels = np.array(self.car.segmask == self.goalID, dtype=np.int32).sum()
        return  viz_pixels/(h*w)

    def push_penalty(self):
        return -1e-4 * self.car.head_force**0.1

    def step(self, action):
        # Feed action to the car and get observation of car's state
        self.car.apply_action(action)
        p.stepSimulation()
        car_ob, cam_ob = self.car.get_observation()

        dist_rew, dist_to_goal = self.dist_reward(car_ob)

        vis_rew = 0#self.visibility_reward() 
        push_pen = self.push_penalty()

        reward = dist_rew + vis_rew + push_pen

        # Done by running off boundaries
        if (car_ob[0] >= 5 or car_ob[0] <= -5 or
                car_ob[1] >= 5 or car_ob[1] <= -5):
            # reward = -10
            self.done = True
        # Done by reaching goal
        elif dist_to_goal < 1:
            self.done = True
            reward = 50

        ob = (np.array(car_ob + self.goal, dtype=np.float32), cam_ob)
        return ob, reward, self.done, {"dist_reward":dist_rew, "visibility_reward": vis_rew, "push_penalty": push_pen}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        # Reload the plane and car
        Plane(self.client)
        p.stepSimulation()
        self.car = Car(self.client)

        # Set the goal to a random target
        x = self.np_random.uniform(2, 4)
        y = self.np_random.uniform(-4, 4)
        self.goal = (x, y)

        # Visual element of the goal
        goal = Goal(self.client, self.goal)
        self.goalID = goal.id

        # Reset obstacles
        Obstacles(self.client)
        p.stepSimulation()
        time.sleep(0.1)

        self.done = False

        # Get observation to return
        car_ob, cam_ob = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                           (car_ob[1] - self.goal[1]) ** 2))
        return np.array(car_ob + self.goal, dtype=np.float32), cam_ob

    def render(self, mode='human'):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))

        # Base information
        car_id, client_id = self.car.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(car_id, client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.pause(.00001)

    def close(self):
        p.disconnect(self.client)
