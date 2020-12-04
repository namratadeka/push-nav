import torch
from torch import nn
import numpy as np

from push_policy.models.network import Network


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_fc = Network(cfg["state_fc"])
        self.cnn = Network(cfg["cnn"])
        self.actor_fc = Network(cfg["actor_fc"])

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
        img_features = self.cnn(image)
        img_features = torch.flatten(img_features, start_dim=1)
        features = torch.cat([state_features, img_features], dim=1)
        action = self.actor_fc(features)

        return action

class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.state_fc = Network(cfg["state_fc"])
        self.cnn = Network(cfg["cnn"])
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
        img_features = self.cnn(image)
        img_features = torch.flatten(img_features, start_dim=1)
        features = torch.cat([state_features, img_features], dim=1)
        value = self.critic_fc(features)

        return value
