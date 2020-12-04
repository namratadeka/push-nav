import os
import gym
import torch
import time
import yaml
import argparse
import pybullet as p
from os.path import join

import push_policy
from ppo import PPO



def main(cfile):
    cfg = yaml.load(open(cfile, "r"), Loader=yaml.FullLoader)
    env = gym.make('PushNav-v0', mode='gui')

    agent = PPO(env, cfg['network'])

    # agent.load_model("agent.pth")
    agent.learn(10000)
    # agent.save_model("agent.pth")

    # env = gym.make('PushNav-v0')
    # ob = env.reset()
    # while True:
    #     p.stepSimulation()
        
    # while True:
    #     action = agent(ob)
    #     ob, _, done, _ = env.step(action)
    #     env.render()
    #     if done:
    #         ob = env.reset()
    #         time.sleep(1/30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        required=True,
        help="name of config file"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu id"
    )
    (args, unknown_args) = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfile = join('./push_policy/config', args.version)
    main(cfile)
