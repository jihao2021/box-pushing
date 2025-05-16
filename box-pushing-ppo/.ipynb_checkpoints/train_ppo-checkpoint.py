import os
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import time

from scripts import utils
from scripts.pygame_env_ppo import CustomEnv
from scripts.PPOMultiAgent import PPOMultiAgent

# Training parameters
num_episodes = 8000
max_epLength = 500

# Default agent settings
NUMBER_OF_AGENT = 5

# PPO hyperparameters
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 2000
K_EPOCHS = 4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5


def run_game():
    # metrics storage
    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    total_distance_list = [[] for _ in range(NUMBER_OF_AGENT)]

    # reset global counters
    env.seed(random_seed)
    observations = env.reset()

    for episode in range(1, num_episodes + 1):
        episode_steps = 0
        episode_rewards = [0.0] * NUMBER_OF_AGENT
        done_flags = {f"Pusher_NN{i+1}": False for i in range(NUMBER_OF_AGENT)}
        observations = env.reset()

        while True:
            # select PPO actions
            actions = ppo_agent.select_actions(observations)

            # step environment
            observations_, rewards, dones, _ = env.step(actions, ppo_agent.n_agents)

            # accumulate rewards
            for i in range(NUMBER_OF_AGENT):
                name = f"Pusher_NN{i+1}"
                episode_rewards[i] += rewards[name]

            # store rewards and dones for PPO
            ppo_agent.store_rewards(rewards, dones)

            observations = observations_
            episode_steps += 1

            # check termination for first agent to end episode
            if dones[f"Pusher_NN1"] or episode_steps >= max_epLength:
                # record distance & rewards
                for idx, pusher in enumerate(env.object_pushers):
                    total_distance_list[idx].append(pusher.distance_travelled)
                for idx in range(NUMBER_OF_AGENT):
                    reward_list[idx].append(episode_rewards[idx])
                break

        # Logging
        avg_reward = np.mean(reward_list[0][-25:]) if episode >= 25 else np.mean(reward_list[0])
        print(f"Episode {episode}, Last25Avg: {avg_reward:.2f}")

    print('Training finished!')

    # Save PPO models
    ppo_agent.save_models()

    # Dump rewards and distances to CSV
    base_dir = f"./ppo_runs/NUM_AGENT{NUMBER_OF_AGENT:02}/SEED{random_seed:04d}"
    reward_dir = os.path.join(base_dir, 'reward_output')
    dist_dir = os.path.join(base_dir, 'distance_output')
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    with open(os.path.join(reward_dir,
                           f'reward_history.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reward_list)

    with open(os.path.join(dist_dir,
                           f'distance_history.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(total_distance_list)

    # Optional: plot losses
    ppo_agent.plot_losses()


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-number_agent', type=int, default=5)
    parser.add_argument('-random_seed', type=int, default=0)
    args = parser.parse_args()

    NUMBER_OF_AGENT = args.number_agent
    random_seed = args.random_seed

    # set seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # setup env & PPO
    env = CustomEnv(random_initialization=False, n_agents=NUMBER_OF_AGENT)
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    ppo_agent = PPOMultiAgent(
        n_agents=NUMBER_OF_AGENT,
        input_dim=input_dim,
        n_actions=n_actions,
        lr=LR,
        gamma=GAMMA,
        eps_clip=EPS_CLIP,
        update_timestep=UPDATE_TIMESTEP,
        k_epochs=K_EPOCHS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        path=f"./ppo_runs/NUM_AGENT{NUMBER_OF_AGENT:02}/SEED{random_seed:04d}/models"
    )

    # initialize observation names to match PPO agent naming
    # PPO expects observations as a dict: {'Pusher_NN1': state1, ...}
    # ensure CustomEnv.step returns rewards/dones keyed by these names

    run_game()
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
