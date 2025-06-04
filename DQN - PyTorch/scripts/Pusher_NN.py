import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt


class MyModel(nn.Module):
    def __init__(self, input_dim, size=16, num_actions=6):
        super(MyModel, self).__init__()
        # Shared dense layer
        self.fc1 = nn.Linear(input_dim, size)
        # Value stream
        self.value = nn.Linear(size, 1)
        # Advantage stream
        self.advantage = nn.Linear(size, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.value(x)
        advantage = self.advantage(x)
        # Combine into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class Pusher_NN:
    def __init__(self,
                 name,
                 n_actions,
                 n_features,
                 load_model,
                 path,
                 output_graph,
                 number_agent,
                 random_seed,
                 n_agents_trained_model,
                 random_seed_trained_model,
                 number_episode
                 ):
        self.name = name
        self.step = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = 0.001
        self.gamma = 0.99
        self.startE = 1.0
        self.replace_target_iter = 200
        self.memory_size = 1000
        self.batch_size = 32
        self.load_model = load_model
        self.path = path
        self.anneling_steps = 100000
        self.epsilon = self.startE
        self.endE = 0.005
        self.stepDrop = (self.startE - self.endE) / self.anneling_steps
        self.pre_train_steps = 2000
        self.dueing = True
        self.dueing_plus_double = False

        self.number_agent = number_agent
        self.random_seed = random_seed
        self.n_agents_trained_model = n_agents_trained_model
        self.random_seed_trained_model = random_seed_trained_model
        self.number_episode = number_episode

        self.learn_step_counter = 0
        # Initialize replay memory
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))
        self.memory_counter = 0
        self.cost_his = []

        # Build networks
        n_l1 = 16
        self.eval_net = MyModel(input_dim=self.n_features, size=n_l1, num_actions=self.n_actions)
        self.target_net = MyModel(input_dim=self.n_features, size=n_l1, num_actions=self.n_actions)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

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

        # Hit counters
        self.one_hit = 0
        self.two_hit = 0
        self.three_hit = 0
        self.four_hit = 0
        self.five_hit = 0
        self.six_hit = 0

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_, done))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self.step >= self.pre_train_steps and self.epsilon > self.endE:
            self.epsilon -= self.stepDrop
        if np.random.uniform() < self.epsilon or self.step < self.pre_train_steps:
            action = np.random.randint(0, self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.eval_net(obs)
            action = int(q_values.argmax().item())
        return action

    def get_qmax(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.eval_net(obs)
        return q_values.max().item()

    def choose_action_greedy(self, observation, epsilon_greedy):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        self.epsilon = epsilon_greedy
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        with torch.no_grad():
            q_values = self.eval_net(obs)
        return int(q_values.argmax().item())

    def choose_action_random(self):
        return np.random.randint(0, self.n_actions)

    def learn(self):
        # Replace target network periodically
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        # Sample batch
        max_mem = min(self.memory_counter, self.memory_size)
        batch_indices = np.random.choice(max_mem, self.batch_size)
        batch = self.memory[batch_indices, :]

        s_batch = torch.tensor(batch[:, :self.n_features], dtype=torch.float32)
        a_batch = batch[:, self.n_features].astype(int)
        r_batch = torch.tensor(batch[:, self.n_features + 1], dtype=torch.float32)
        s_next = torch.tensor(batch[:, - (self.n_features + 1):-1], dtype=torch.float32)
        done_batch = torch.tensor(batch[:, -1], dtype=torch.float32)

        # Compute Q-values
        q_eval = self.eval_net(s_batch)
        q_next = self.target_net(s_next)

        if self.dueing_plus_double:
            q_eval_next = self.eval_net(s_next)
            max_actions = q_eval_next.argmax(dim=1)
            selected_q_next = q_next.gather(1, max_actions.unsqueeze(1)).squeeze(1)
        else:
            selected_q_next = q_next.max(dim=1)[0]

        # Build target tensor
        q_target = q_eval.clone().detach()
        q_target[range(self.batch_size), a_batch] = r_batch + self.gamma * selected_q_next * (1 - done_batch)

        # Optimize
        loss = F.mse_loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def loading_self(self, others_number, others_random_seed):
        print('Loading Model...')
        load_path = f"./NUM_AGENT{others_number:02}/{others_random_seed:04d}/saved_model"
        eval_file = os.path.join(load_path, f"model-{self.number_episode}{self.name}-eval.pth")
        target_file = os.path.join(load_path, f"model-{self.number_episode}{self.name}-target.pth")
        self.eval_net.load_state_dict(torch.load(eval_file))
        self.target_net.load_state_dict(torch.load(target_file))

    def loading_others_higher(self, others_number, others_random_seed):
        # Same as loading_self for higher count
        self.loading_self(others_number, others_random_seed)

    def loading_others_lower(self, others_number, others_random_seed):
        print('Loading Model...')
        load_path = f"./NUM_AGENT{others_number:02}/{others_random_seed:04d}/saved_model"
        # Build list of replacement names
        mine = [f"Pusher_NN{i+1}" for i in range(self.number_agent)]
        lower = [f"Pusher_NN{i+1}" for i in range(others_number)]
        lower = (lower * ((self.number_agent // others_number) + 1))[:self.number_agent]
        for i, nm in enumerate(mine):
            if self.name == nm:
                lbl = lower[i]
                eval_file = os.path.join(load_path, f"model-{self.number_episode}{lbl}-eval.pth")
                target_file = os.path.join(load_path, f"model-{self.number_episode}{lbl}-target.pth")
                self.eval_net.load_state_dict(torch.load(eval_file))
                self.target_net.load_state_dict(torch.load(target_file))
