import numpy as np
import torch
from .Pusher_PPO import PusherPPO

class PPOMultiAgent:
    def __init__(
            self,
            n_agents,
            input_dim,
            n_actions,
            lr=3e-4,
            gamma=0.99,
            eps_clip=0.2,
            update_timestep=2000,
            k_epochs=4,
            entropy_coef=0.01,
            value_coef=0.5,
            path='./'
    ):
        self.n_agents = n_agents
        self.step = 0
        # create one PPO agent per pusher
        self.agents = []
        for i in range(n_agents):
            agent = PusherPPO(
                input_dim=input_dim,
                num_actions=n_actions,
                lr=lr,
                gamma=gamma,
                eps_clip=eps_clip,
                update_timestep=update_timestep,
                k_epochs=k_epochs,
                entropy_coef=entropy_coef,
                value_coef=value_coef,
                path=path
            )
            self.agents.append(agent)
        # shared timestep counter for batched update
        self.global_timestep = 0

    def save_models(self):
        for idx, agent in enumerate(self.agents):
            agent.save(f"ppo_agent{idx}_model.pth")

    def load_models(self):
        for idx, agent in enumerate(self.agents):
            agent.load(f"ppo_agent{idx}_model.pth")

    def select_actions(self, observations):
        """
        observations: dict mapping 'Pusher_NN{i+1}' to state vectors
        returns: dict mapping agent names to actions
        """
        actions = {}
        for i, agent in enumerate(self.agents):
            name = f"Pusher_NN{i+1}"
            state = observations[name]
            act = agent.select_action(state)
            actions[name] = act
        return actions

    def store_rewards(self, rewards, dones):
        """
        rewards, dones: dicts mapping agent names to scalar values
        """
        for i, agent in enumerate(self.agents):
            name = f"Pusher_NN{i+1}"
            agent.store_reward(rewards[name], dones[name])
            # update global timestep
            self.global_timestep += 1

    def plot_losses(self):
        import matplotlib.pyplot as plt
        for idx, agent in enumerate(self.agents):
            plt.plot(agent.loss_history, label=f"Agent{idx+1}")
        plt.xlabel('Update step')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
