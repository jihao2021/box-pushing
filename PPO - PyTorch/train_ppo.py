import os
import sys
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import time

# scripts_path = os.path.join(os.path.dirname(__file__), "scripts")
# if scripts_path not in sys.path:
#     sys.path.append(scripts_path)

from scripts import utils
from scripts.pygame_env import CustomEnv
from scripts.PPO import PPO

# Training parameters
num_episodes = 16000
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
    ppo_agent.step = 0

    # Reset each Pusher_NN's internal step counter
    for p in Pusher_NN_list:
        p.step = 0

    for episode in range(1, num_episodes + 1):
        observations = env.reset()
        pusher_list = env.object_pushers[:]

        episode_steps = 0
        episode_rewards = [0.0] * NUMBER_OF_AGENT
        # done_flags = {f"Pusher_NN{i+1}": False for i in range(NUMBER_OF_AGENT)}

        while True:
            # select PPO actions
            actions = ppo_agent.choose_action(observations)

            # step environment
            # print("Actions in Train PPO:", actions)
            actions_only = {k: v[0] for k, v in actions.items()}
            observations_, rewards, done, _ = env.step(actions_only, ppo_agent.Pusher_NN_name)

            # accumulate rewards
            for i, name in enumerate(ppo_agent.Pusher_NN_name):
                episode_rewards[i] += rewards[name]

            # store rewards and dones for PPO
            ppo_agent.store_transition(observations, actions, rewards, observations_, done)

            for p in Pusher_NN_list:
                if p.step > p.pre_train_steps and p.step % 5 == 0:
                    p.learn()

            observations = observations_
            episode_steps += 1
            ppo_agent.step += 1
            for p in Pusher_NN_list:
                p.step += 1

            # check termination for first agent to end episode
            if done[ppo_agent.Pusher_NN_name[0]] or episode_steps >= max_epLength:
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
    # ppo_agent.plot_losses()


if __name__ == "__main__":
    start_time = time.time()
    loading_model = False

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

    ppo_agent = PPO(
        n_agents=NUMBER_OF_AGENT,
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        load_model=loading_model,
        path=f"./ppo_runs/NUM_AGENT{NUMBER_OF_AGENT:02}/SEED{random_seed:04d}/models",
        output_graph=True,
        random_seed=random_seed,
        n_agents_trained_model=0,
        random_seed_trained_model=0,
        number_episode=num_episodes
        )

    # initialize observation names to match PPO agent naming
    # PPO expects observations as a dict: {'Pusher_NN1': state1, ...}
    # ensure CustomEnv.step returns rewards/dones keyed by these names

    Pusher_NN_list = ppo_agent.Pusher_NN

    run_game()
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
