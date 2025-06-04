import os
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
import torch
import csv
import time

from scripts import utils
from scripts.pygame_env import CustomEnv
from scripts.DQN import DeepQNetwork

# Training parameters
num_episodes = 8000
max_epLength = 500

# Default agent settings
NUMBER_OF_AGENT = 5
NUMBER_OF_AGENT_TRAINED_MODEL = 0


def run_game():
    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    RL.step = 0
    # Reset each Pusher_NN's internal step counter
    for p in Pusher_NN_list:
        p.step = 0

    total_distance_list = [[] for _ in range(NUMBER_OF_AGENT)]

    for episode in range(1, num_episodes + 1):
        observation = env.reset()
        pusher_list = env.object_pushers[:]

        episode_steps = 0
        episode_reward_list = [0] * NUMBER_OF_AGENT

        while True:
            # Choose actions
            action = RL.choose_action(observation)

            # Step environment
            observation_, reward, done, _ = env.step(action, RL.Pusher_NN_name)

            # Accumulate reward
            for i, name in enumerate(RL.Pusher_NN_name):
                episode_reward_list[i] += reward[name]

            # Store transition
            RL.store_transition(observation, action, reward, observation_, done)

            # Learning step for each agent
            for p in Pusher_NN_list:
                if p.step > p.pre_train_steps and p.step % 5 == 0:
                    p.learn()

            observation = observation_
            episode_steps += 1
            RL.step += 1
            for p in Pusher_NN_list:
                p.step += 1

            # End condition
            if done[RL.Pusher_NN_name[0]] or episode_steps > max_epLength:
                for i, p in enumerate(pusher_list):
                    total_distance_list[i].append(p.distance_travelled)
                for i in range(NUMBER_OF_AGENT):
                    reward_list[i].append(episode_reward_list[i])
                break

        # Logging
        print(f"Episode {episode}, epsilon: {Pusher_NN_list[0].epsilon:.4f}")
        if episode % 25 == 0:
            avg_reward = np.mean(reward_list[0][-25:])
            print(f"Last 25 avg reward: {avg_reward:.2f}, epsilon: {Pusher_NN_list[0].epsilon:.4f}")

    print('Training finished!')
    RL.save_models()

    # Create output dirs
    base_dir = f"./DQN/NUM_AGENT{NUMBER_OF_AGENT:02}/{random_seed:04d}"
    reward_dir = os.path.join(base_dir, 'reward_output')
    dist_dir = os.path.join(base_dir, 'distance_output')
    os.makedirs(reward_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    # Dump rewards
    with open(os.path.join(reward_dir,
                           f'reward_history-NUM_AGENT{NUMBER_OF_AGENT:02}-SEED{random_seed:04d}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reward_list)

    # Dump distances
    with open(os.path.join(dist_dir,
                           f'distance_history-NUM_AGENT{NUMBER_OF_AGENT:02}-SEED{random_seed:04d}.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(total_distance_list)


if __name__ == "__main__":
    start_time = time.time()
    loading_model = False

    # CLI args or defaults
    parser = argparse.ArgumentParser()
    parser.add_argument('-number_agent', type=int, default=10)
    parser.add_argument('-random_seed', type=int, default=5)
    parser.add_argument('-number_agent_trained_model', type=int, default=1)
    args = parser.parse_args()

    NUMBER_OF_AGENT = args.number_agent
    random_seed = args.random_seed
    NUMBER_OF_AGENT_TRAINED_MODEL = args.number_agent_trained_model

    # Set seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    print(f"Number of agents: {NUMBER_OF_AGENT}, Seed: {random_seed}")

    env = CustomEnv(random_initialization=False, n_agents=NUMBER_OF_AGENT)
    RL = DeepQNetwork(
        n_agents=NUMBER_OF_AGENT,
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        load_model=loading_model,
        path=f"./NUM_AGENT{NUMBER_OF_AGENT:02}/{random_seed:04d}/saved_model",
        output_graph=True,
        random_seed=random_seed,
        n_agents_trained_model=NUMBER_OF_AGENT_TRAINED_MODEL,
        random_seed_trained_model=None,
        number_episode=num_episodes
    )

    # Shortcut to agent list
    Pusher_NN_list = RL.Pusher_NN

    print("Initial action example:", RL.choose_action(env.reset()))
    print("Pusher_NN names:", RL.Pusher_NN_name)

    run_game()
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
