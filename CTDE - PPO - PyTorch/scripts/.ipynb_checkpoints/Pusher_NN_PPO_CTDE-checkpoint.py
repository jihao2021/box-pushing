import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim, hidden_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

class Pusher_NN_PPO_CTDE:
    def __init__(self, name, n_features, n_actions, centralized, load_model, path, output_graph, n_agents,
                 random_seed, n_agents_trained_model, random_seed_trained_model, number_episode,
                 lr=3e-4, hidden_size=64):
        self.name = name
        self.n_features = n_features
        self.n_actions = n_actions
        self.global_state_dim = n_agents * n_features
        self.centralized = centralized

        self.load_model=load_model
        self.path=path
        self.output_graph=output_graph
        self.number_agent=n_agents
        self.random_seed=random_seed
        self.n_agents_trained_model=n_agents_trained_model
        self.random_seed_trained_model=random_seed_trained_model
        self.number_episode=number_episode

        self.pre_train_steps = 2000

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coeff = 0.01
        self.batch_size = 64

        self.actor = Actor(n_features, hidden_size, n_actions)
        self.critic = CentralizedCritic(self.global_state_dim, hidden_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = []

    def choose_action(self, obs):  # obs is local_obs
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def store_transition(self, obs, global_obs, action, logp, reward, next_global_obs, done):
        self.memory.append((obs, global_obs, action, logp, reward, next_global_obs, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        obs, global_obs, a, logp_old, r, next_global_obs, done = zip(*self.memory)
        self.memory = []

        obs = torch.tensor(obs, dtype=torch.float32)
        global_obs = torch.tensor(global_obs, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        logp_old = torch.tensor(logp_old, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        next_global_obs = torch.tensor(next_global_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        with torch.no_grad():
            v_next = self.critic(next_global_obs)
            v_now = self.critic(global_obs)
            td_target = r + self.gamma * v_next * (1 - done)
            advantage = td_target - v_now
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(5):
            logits = self.actor(obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(a)
            entropy = dist.entropy()

            ratio = torch.exp(logp - logp_old)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(self.critic(global_obs), td_target)
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()