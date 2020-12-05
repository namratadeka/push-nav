import os
import gym
import torch
import wandb
import time
import yaml
import argparse
import pybullet as p
from os.path import join, splitext

import push_policy
from ppo import PPO


def init_wandb(cfile) -> None:
    wandb.init(project="push-nav", name=splitext(cfile)[0])

def main(cfile, mode, use_wandb):
    if use_wandb:
        init_wandb(cfile)
    cfg = yaml.load(open(cfile, "r"), Loader=yaml.FullLoader)
    env = gym.make('PushNav-v0', mode=mode)

    agent = PPO(env, cfg['network'], use_wandb)

    # agent.load_model("agent.pth")
    agent.learn(1000000)
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
        "-m",
        "--mode",
        type=str,
        default="direct",
        help="simulation mode (direct or gui)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu id"
    )
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="log to wandb"
    )
    (args, unknown_args) = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfile = join('./push_policy/config', args.version)
    main(cfile, args.mode, args.wandb)
