import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .Pusher_NN_PPO import Pusher_NN_PPO

class PPO:
    def __init__(self, n_agents, n_actions, n_features, load_model, path,
                 output_graph, random_seed, n_agents_trained_model, random_seed_trained_model, number_episode):
        self.num_episodes = number_episode
        self.step = 0
        self.n_agents = n_agents
        self.Pusher_NN_name = [f"Pusher_NN{i+1}" for i in range(n_agents)]
        self.Pusher_NN = [
            Pusher_NN_PPO(name=self.Pusher_NN_name[i],
                      n_actions=n_actions,
                      n_features=n_features,
                      load_model=load_model,
                      path=path,
                      output_graph=output_graph,
                      number_agent=n_agents,
                      random_seed=random_seed,
                      n_agents_trained_model=n_agents_trained_model,
                      random_seed_trained_model=random_seed_trained_model,
                      number_episode=number_episode)
            for i in range(n_agents)]
        
    def save_models(self):
        """
        Save weights to `models/job{n_agents}` directory using PyTorch.
        """
        parent_dir = f"models/job{self.n_agents}/"
        os.makedirs(parent_dir, exist_ok=True)

        for i in range(self.n_agents):
            eval_path = os.path.join(parent_dir, f"model-{self.num_episodes}Pusher_NN{i}-eval.pth")
            target_path = os.path.join(parent_dir, f"model-{self.num_episodes}Pusher_NN{i}-target.pth")
            torch.save(self.Pusher_NN[i].policy.state_dict(), eval_path)
            torch.save(self.Pusher_NN[i].optimizer.state_dict(), target_path)

    def loading_parameters(self, num_trained, random_seed):
        """
        Load pretrained weights for a team of agents, sampling if needed.
        """
        if num_trained == self.n_agents:
            print(f"loading exact {num_trained} trained models")
            for i in range(self.n_agents):
                eval_path = f"models/job{self.n_agents}/model-{self.num_episodes}Pusher_NN{i}-eval.pth"
                target_path = f"models/job{self.n_agents}/model-{self.num_episodes}Pusher_NN{i}-target.pth"
                self.Pusher_NN[i].eval_net.load_state_dict(torch.load(eval_path))
                self.Pusher_NN[i].target_net.load_state_dict(torch.load(target_path))

        elif num_trained < self.n_agents:
            np.random.seed(random_seed)
            idx = list(range(num_trained))
            sampled_idx = list(np.random.choice(idx, self.n_agents - num_trained))
            idx = idx + sampled_idx

            print(f"sampling from {num_trained} trained models to fit team of {self.n_agents}")
            for i, id_ in enumerate(idx):
                eval_path = f"models/job{num_trained}/model-{self.num_episodes}Pusher_NN{id_}-eval.pth"
                target_path = f"models/job{num_trained}/model-{self.num_episodes}Pusher_NN{id_}-target.pth"
                self.Pusher_NN[i].eval_net.load_state_dict(torch.load(eval_path))
                self.Pusher_NN[i].target_net.load_state_dict(torch.load(target_path))

        else:
            # num_trained > self.n_agents
            np.random.seed(random_seed)
            idx = list(range(num_trained))
            sampled_idx = list(np.random.choice(idx, self.n_agents))
            idx = sampled_idx

            print(f"sampling from {num_trained} trained models to fit team of {self.n_agents}")
            for i, id_ in enumerate(idx):
                eval_path = f"models/job{num_trained}/model-{self.num_episodes}Pusher_NN{id_}-eval.pth"
                target_path = f"models/job{num_trained}/model-{self.num_episodes}Pusher_NN{id_}-target.pth"
                self.Pusher_NN[i].eval_net.load_state_dict(torch.load(eval_path))
                self.Pusher_NN[i].target_net.load_state_dict(torch.load(target_path))

    def choose_action(self, observation):
        actions = {}
        for i in range(self.n_agents):
            obs = observation[self.Pusher_NN_name[i]]
            action, logp, ent = self.Pusher_NN[i].choose_action(obs)
            actions[self.Pusher_NN_name[i]] = (action, logp, ent)
        return actions

    def store_transition(self, s, a, r, s_, done):
        for i in range(self.n_agents):
            name = self.Pusher_NN_name[i]
            action, logp, ent = a[name]
            self.Pusher_NN[i].store_transition(s[name], action, logp, r[name], s_[name], done[name], 0, ent)

    def learn(self):
        for i in range(self.n_agents):
            self.Pusher_NN[i].learn()
