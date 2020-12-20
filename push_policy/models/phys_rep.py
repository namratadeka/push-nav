import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from push_policy.models.network import Network


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_fc = Network(cfg["state_fc"])
        self.phys_encoder = Network(cfg["phys_encoder"])
        self.policy_cnn = Network(cfg["policy_cnn"])
        self.actor_fc = Network(cfg["actor_fc"])
        self.action_encoder = Network(cfg["action_encoder"])
        self.motion_predictor = Network(cfg["motion_predictor"])

    def cross_convolve(self, input, kernels):
        C = input.shape[1]
        result = list()
        for i in range(C):
            result.append(F.conv2d(input[:, i:i+1], kernels[:, i:i+1]))

        return torch.cat(result, dim=1)

    def forward(self, state, image):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        state_features = self.state_fc(state)
        phys_features = self.phys_encoder(image)
        policy_features = self.policy_cnn(phys_features)
        policy_features = torch.flatten(policy_features, start_dim=1)
        features = torch.cat([state_features, policy_features], dim=1)
        action = self.actor_fc(features)

        action_features = self.action_encoder(action)
        action_kernels = action_features.reshape(-1, 32, 5, 5)

        state_action_ft = self.cross_convolve(phys_features, action_kernels)
        flow = self.motion_predictor(state_action_ft)

        return action, flow

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.state_fc = Network(cfg["state_fc"])
        self.policy_cnn = Network(cfg["policy_cnn"])
        self.critic_fc = Network(cfg["critic_fc"])

    def forward(self, state, image):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2)

        state_features = self.state_fc(state)
        img_features = self.policy_cnn(image)
        img_features = torch.flatten(img_features, start_dim=1)
        features = torch.cat([state_features, img_features], dim=1)
        value = self.critic_fc(features)

        return value
