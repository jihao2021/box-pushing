import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os

class PPOModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64, num_actions=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.policy_head(x), self.value_head(x)

class PusherPPO:
    def __init__(self,
                 input_dim,
                 num_actions,
                 lr=3e-4,
                 gamma=0.99,
                 eps_clip=0.2,
                 update_timestep=2000,
                 k_epochs=4,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 path='./'):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_timestep = update_timestep
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.path = path

        self.model = PPOModel(input_dim, hidden_size=64, num_actions=num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'rewards': [],
            'dones': []
        }
        self.timestep = 0
        self.loss_history = []

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)

        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(logprob)
        return action.item()

    def store_reward(self, reward, done):
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.timestep += 1
        if self.timestep % self.update_timestep == 0:
            self.update()
            self.clear_memory()

    def update(self):
        # compute discounted returns
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).detach()
        # convert lists to tensors
        old_states = torch.cat(self.memory['states'])
        old_actions = torch.tensor(self.memory['actions']).detach()
        old_logprobs = torch.stack(self.memory['logprobs']).detach()

        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # PPO update
        for _ in range(self.k_epochs):
            logits, state_values = self.model(old_states)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(old_actions)            
            entropy = dist.entropy().mean()

            # ratios
            ratios = torch.exp(new_logprobs - old_logprobs)
            advantages = returns - state_values.squeeze().detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())

    def clear_memory(self):
        for key in self.memory:
            self.memory[key] = []

    def save(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.path, filename))

    def load(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.path, filename)))

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.show()
