import csv
import numpy as np
import matplotlib.pyplot as plt
import constant  # Ensure this module exists and color_list is defined.

def plot_reward(file_path):
    NUMBER_OF_AGENT = count_rows_csv(file_path)
    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    color_list = constant.color_list

    with open(file_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]  # Read all rows at once

        for i in range(NUMBER_OF_AGENT):
            if len(list1) > i:  # Ensure there is enough data
                reward_list[i] = list(map(float, list1[i]))
            if len(list1) > i + NUMBER_OF_AGENT:  # Ensure there is enough data
                additional_rewards = list(map(float, list1[i + NUMBER_OF_AGENT]))
                reward_list[i] = [(x + y) / 2 for x, y in zip(reward_list[i], additional_rewards)]

        combined_rewards = np.mean(reward_list, axis=0)

        window_size = 100  # Adjust the window size for more smoothing
        window = np.ones(window_size) / window_size
        smoothed_rewards = np.convolve(combined_rewards, window, mode='valid')

        plt.plot(smoothed_rewards, color='blue', label='Smoothed Average Reward')

        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('Training Episode (every 100 episodes)')
        plt.show()

def count_rows_csv(file_path):
    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            row_count = sum(1 for row in reader)
        return row_count
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

file_path = './training_output/reward_output/reward150.csv'
plot_reward(file_path)