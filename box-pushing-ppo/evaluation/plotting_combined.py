import csv
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append("..")
from scripts import utils

def read_csv_data(file_path, number_of_agents):
    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data = [list(map(float, row)) for row in readCSV]
    assert len(data) == number_of_agents
    return data

def average_across_seeds(base_path, file_pattern, number_of_agents, num_seeds):
    accumulated = None

    for seed in range(1, num_seeds + 1):
        file_path = os.path.join(base_path, file_pattern.format(seed=str(seed).zfill(4)))
        data = read_csv_data(file_path, number_of_agents)
        
        if accumulated is None:
            accumulated = [np.array(agent_data) for agent_data in data]
        else:
            for i in range(number_of_agents):
                accumulated[i] += np.array(data[i])

    avg_data = [agent_data / num_seeds for agent_data in accumulated]
    return avg_data

def smooth_data(data_list):
    return [np.convolve(data, np.ones((2,)) / 2, mode='valid') for data in data_list]

def plot_data(data_list, ylabel, title, filename, number_of_agents):
    color_list = utils.color_list
    label_list = ['agent' + str(i+1) for i in range(number_of_agents)]

    for i in range(number_of_agents):
        plt.plot(data_list[i], color_list[i], label=label_list[i])

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel('Training Episode (every 100 episode)')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def plot_all(NUMBER_OF_AGENT, NUM_SEEDS):
    reward_avg = average_across_seeds(
        base_path='../ppo_runs/NUM_AGENT05/',
        file_pattern='SEED{seed}/reward_output/reward_history.csv',
        number_of_agents=NUMBER_OF_AGENT,
        num_seeds=NUM_SEEDS
    )
    reward_smoothed = smooth_data(reward_avg)
    plot_data(reward_smoothed, 'Reward', 'Average Reward per Agent', 'reward_plot_avg.png', NUMBER_OF_AGENT)

    distance_avg = average_across_seeds(
        base_path='../ppo_runs/NUM_AGENT05/',
        file_pattern='SEED{seed}/distance_output/distance_history.csv',
        number_of_agents=NUMBER_OF_AGENT,
        num_seeds=NUM_SEEDS
    )
    distance_smoothed = smooth_data(distance_avg)
    plot_data(distance_smoothed, 'Total Step Sizes', 'Average Total Distance per Agent', 'total_distance_plot_avg.png', NUMBER_OF_AGENT)

# Run it
plot_all(NUMBER_OF_AGENT=5, NUM_SEEDS=100)