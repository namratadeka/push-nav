import wandb
import torch
import numpy as np
from os.path import join
from datetime import datetime
from collections import defaultdict
from torch.distributions import MultivariateNormal
from push_policy.models.actor_critic import Actor, Critic


class PPO:
    def __init__(self, env, network_cfg, use_wandb, outdir):
        self.use_wandb = use_wandb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_hyperparameters()
        
        self.env = env
        self.act_dim = env.action_space.shape[0]

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)

        self.actor = Actor(network_cfg).to(self.device)
        self.critic = Critic(network_cfg).to(self.device)
        if use_wandb:
            wandb.watch(self.actor)
            wandb.watch(self.critic)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=2.5e-6)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.now = datetime.now().strftime("%D-%H-%M").replace('/','-')
        self.outdir = outdir

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2000
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.9
        self.epochs = 4
        self.clip = 0.2
        self.entropy_beta = 0.1
        self.minibatch_size = 256

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
        batch_info = defaultdict(list)

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_info = defaultdict(list)
            state_obs, img_obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs_state.append(state_obs)
                batch_obs_img.append(img_obs)

                action, log_prob = self.get_action(state_obs, img_obs)
                for k in range(4):
                    obs, rew, done, info = self.env.step(action)
                state_obs, img_obs = obs[0], obs[1]

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                ep_info['dist_reward'].append(info['dist_reward'])
                ep_info['visibility_reward'].append(info['visibility_reward'])
                ep_info['push_penalty'].append(info['push_penalty'])

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_info['dist'].append(ep_info['dist_reward'])
            batch_info['vis'].append(ep_info['visibility_reward'])
            batch_info['push'].append(ep_info['push_penalty'])

        batch_obs_state = torch.tensor(batch_obs_state, dtype=torch.float).to(self.device)
        batch_obs_img = torch.tensor(batch_obs_img, dtype=torch.float).to(self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)
        for key in batch_info:
            batch_info[key] = self.compute_rtgs(batch_info[key])

        return batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_info

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
            batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_info = self.rollout()

            avg_rew = self.avg_reward_per_episode(batch_rtgs, batch_lens)
            avg_info = dict()
            for key in batch_info:
                avg_info[key] = self.avg_reward_per_episode(batch_info[key], batch_lens)
            print("[{}] Average episodic reward: {}".format(itr, avg_rew))
            if self.use_wandb:
                wandb.log({"Average episodic reward": avg_rew}, step=itr)
                wandb.log(avg_info, step=itr)

            t_so_far += np.sum(batch_lens)
            
            batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs = self.randomize(batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs)
            # losses = defaultdict(list)
            for i in range(self.epochs):
                for k in range(0, batch_rtgs.shape[0], self.minibatch_size):

                    obs_state = batch_obs_state[k : k + self.minibatch_size]
                    obs_img = batch_obs_img[k : k + self.minibatch_size]
                    acts = batch_acts[k : k + self.minibatch_size]
                    rtgs = batch_rtgs[k : k + self.minibatch_size]
                    log_probs = batch_log_probs[k : k + self.minibatch_size]

                    V, _ = self.evaluate(obs_state, obs_img, acts)

                    A_k = rtgs - V.detach()
                    del V

                    # normalize advantages
                    A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

                    V, curr_log_probs = self.evaluate(obs_state, obs_img, acts)

                    ratios = torch.exp(curr_log_probs - log_probs)

                    # surrogate losses
                    surr1 = ratios * A_k
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                    entropy = self.entropy_beta * (-(torch.exp(curr_log_probs) * curr_log_probs)).mean()

                    actor_loss = (-torch.min(surr1, surr2)).mean() - entropy
                    critic_loss = torch.nn.MSELoss()(V, rtgs)

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()

            #     if self.use_wandb:
            #         losses['actor_loss'].append(actor_loss.item())
            #         losses['critic_loss'].append(critic_loss.item())
            #         losses['entropy'].append(entropy.item())

            # if self.use_wandb:
            #     for key in losses:
            #         losses[key] = np.mean(losses[key])
            #     wandb.log(losses, step=itr)
            if itr % 100 == 0:
                self.save_model(itr)
            itr += 1

    def randomize(self, batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs):
        idx = np.random.randint(0, batch_rtgs.shape[0], batch_rtgs.shape[0])
        batch_obs_state = batch_obs_state[idx]
        batch_obs_img = batch_obs_img[idx]
        batch_acts = batch_acts[idx]
        batch_log_probs = batch_log_probs[idx]
        batch_rtgs = batch_rtgs[idx]

        return batch_obs_state, batch_obs_img, batch_acts, batch_log_probs, batch_rtgs

    def avg_reward_per_episode(self, batch_rtgs, batch_lens):
        episodic_rewards = []
        for i, ep_len in enumerate(batch_lens):
            total_time_so_far = int(np.sum(batch_lens[:i]))
            episodic_rewards.append(batch_rtgs[total_time_so_far:total_time_so_far+ep_len].sum().item())
        
        return np.mean(episodic_rewards)

    def save_model(self, itr):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, join(self.outdir, self.now, 'iteration_{}.pth'.format(itr)))

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
