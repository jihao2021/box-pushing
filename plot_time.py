import csv
import numpy as np
import matplotlib.pyplot as plt
import constant  # Ensure this module exists and color_list is defined.

def plot_reward(file):
    file_path1 = f"./training_output/reward_output/reward{file}"
    file_path2 = f"./training_output/time_output/time{file}"

    NUMBER_OF_AGENT = count_rows_csv(file_path1)
    if NUMBER_OF_AGENT is None:
        return  # Exit if there's an error in counting rows

    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    time_list = []
    color_list = constant.color_list

    # Read rewards
    with open(file_path1) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]  # Read all rows at once
        for i in range(NUMBER_OF_AGENT):
            if len(list1) > i:  # Ensure there is enough data
                reward_list[i] = list(map(float, list1[i]))

    # Read time
    with open(file_path2) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]  # Read all rows at once
        time_list = list(map(float, list1[0]))  # Convert time values to float

    # Combine rewards and smooth
    combined_rewards = np.mean(reward_list, axis=0)
    window_size = 100  # Adjust the window size for smoothing
    window = np.ones(window_size) / window_size
    smoothed_rewards = np.convolve(combined_rewards, window, mode='valid')

    # Adjust time for smoothed rewards
    valid_time = time_list[window_size-1:]

    # Plot
    plt.plot(valid_time, smoothed_rewards, marker='o', linestyle='-', color='b', label='Smoothed Average Reward')

    # Add title and labels
    plt.title('Reward Over Time')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()

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

file = '15.csv'
plot_reward(file)
