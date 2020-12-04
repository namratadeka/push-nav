import torch
import numpy as np
from torch.distributions import MultivariateNormal
from push_policy.models import Actor, Critic


class PPO:
    def __init__(self, env, network_cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_hyperparameters()
        
        self.env = env
        self.act_dim = env.action_space.shape[0]

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.actor = Actor(network_cfg).to(self.device)
        self.critic = Critic(network_cfg).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 400
        self.max_timesteps_per_episode = 150
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005

    def get_action(self, state, img):
        mean = self.actor(state, img).squeeze()
        dist = MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), log_prob.detach()

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs

    def rollout(self):
        # batch data
        batch_obs_state = []                # batch states
        batch_obs_img = []                  # batch images
        batch_acts = []                     # batch actions
        batch_log_probs = []                # batch log prob of each action
        batch_rews = []                     # batch rewards
        batch_rtgs = []                     # batch rewards-to-go
        batch_lens = []                     # episodic lengths in batch

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            state_obs, img_obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs_state.append(state_obs)
                batch_obs_img.append(img_obs)

                action, log_prob = self.get_action(state_obs, img_obs)
                obs, rew, done, _ = self.env.step(action)
                state_obs, img_obs = obs[0], obs[1]

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs_state = torch.tensor(batch_obs_state, dtype=torch.float).to(self.device)
        batch_obs_img = torch.tensor(batch_obs_img, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def evaluate(self, state, img, acts):
        V = self.critic(state, img).squeeze()

        mean = self.actor(state, img).squeeze()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(acts)

        return V, log_probs

    def learn(self, total_timesteps):
        t_so_far = 0
        itr = 0
        while t_so_far < total_timesteps:
            batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            print("[{}] Average episodic reward: {}".format(itr, batch_rtgs.mean().item()))
            itr += 1
            t_so_far += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs_state, batch_obs_img, batch_acts)

            A_k = batch_rtgs - V.detach()
            del V

            # normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for i in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs_state, batch_obs_img, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
