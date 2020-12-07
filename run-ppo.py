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

def main(cfile, render_mode, use_wandb, mode, load_path=None):
    if use_wandb:
        init_wandb(cfile)
    cfg = yaml.load(open(cfile, "r"), Loader=yaml.FullLoader)
    env = gym.make('PushNav-v0', mode=render_mode)

    agent = PPO(env, cfg['network'], use_wandb, outdir='/data/namrata/models/push-nav')
    if load_path is not None:
        agent.load_model(load_path)

    if mode.__eq__('train'):
        agent.learn(1000000)
        agent.save_model('final')
    else:
        ob = env.reset()
        while True:
            state, cam = ob
            action, _ = agent.get_action(state, cam)
            for k in range(4):
                ob, _, done, _ = env.step(action)
            if done:
                ob = env.reset()
                time.sleep(1/30)


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
        "-sm",
        "--sim_mode",
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
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="train",
        help="policy mode (train or test)"
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="path to saved model"
    )
    (args, unknown_args) = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfile = join('./push_policy/config', args.version)
    main(cfile, args.sim_mode, args.wandb, args.mode, args.load)
