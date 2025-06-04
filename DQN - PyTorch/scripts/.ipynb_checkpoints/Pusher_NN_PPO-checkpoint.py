import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_size=16, num_actions=7):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.policy_logits = nn.Linear(hidden_size, num_actions)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.policy_logits(x), self.value(x)

class Pusher_NN_PPO:
    def __init__(self, name, n_actions, n_features, load_model, path, output_graph,
                 number_agent, random_seed, n_agents_trained_model, random_seed_trained_model, number_episode):
        self.name = name
        self.n_actions = n_actions
        self.n_features = n_features
        self.load_model = load_model
        self.path = path
        self.number_agent = number_agent,
        self.random_seed = random_seed
        self.n_agents_trained_model = n_agents_trained_model
        self.random_seed_trained_model = random_seed_trained_model
        self.number_episode = number_episode
        self.pre_train_steps = 2000

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.lr = 3e-4
        self.name = name
        self.memory = []
        self.batch_size = 64
        self.entropy_coeff = 0.01

        self.policy = Policy(input_dim=n_features, hidden_size=16, num_actions=n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        if not self.load_model:
            print('Error in initializing parameters in Pusher_NN')
        else:
            # Load pretrained parameters as per configuration
            if self.number_agent < self.n_agents_trained_model:
                self.loading_others_higher(self.n_agents_trained_model, self.random_seed_trained_model)
            elif self.number_agent > self.n_agents_trained_model:
                self.loading_others_lower(self.n_agents_trained_model, self.random_seed_trained_model)
            elif self.number_agent == self.n_agents_trained_model:
                self.loading_self(self.n_agents_trained_model, self.random_seed_trained_model)
            else:
                print('Error in loading model in Pusher_NN')

        # # Hit counters
        # self.one_hit = 0
        # self.two_hit = 0
        # self.three_hit = 0
        # self.four_hit = 0
        # self.five_hit = 0
        # self.six_hit = 0

    def choose_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy(obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def store_transition(self, s, a, logp, r, s_, done, v, ent):
        self.memory.append((s, a, logp, r, s_, done, v, ent))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        s, a, logp, r, s_, done, v, ent = zip(*self.memory)
        self.memory = []

        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        logp = torch.tensor(logp, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        s_ = torch.tensor(s_, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)

        with torch.no_grad():
            _, next_v = self.policy(s_)
            next_v = next_v.squeeze()
            td_target = r + self.gamma * next_v * (1 - done)
            advantage = td_target - v

        for _ in range(5):
            logits, values = self.policy(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_logp = dist.log_prob(a)
            ratio = torch.exp(new_logp - logp)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(), td_target)
            entropy_loss = -dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss + self.entropy_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        