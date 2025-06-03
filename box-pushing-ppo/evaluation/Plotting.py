import csv
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from scripts import utils

def plot_reward(NUMBER_OF_AGENT):
    with open('../NUM_AGENT05/0005/reward_output/reward_history-NUM_AGENT05-Random_seed0005.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]

        color_list = utils.color_list
        label_list = ['agent' + str(i+1) for i in range(NUMBER_OF_AGENT)]

        reward_list = [list(map(float, list1[i])) for i in range(NUMBER_OF_AGENT)]

        result = [np.convolve(reward, np.ones((2,)) / 2, mode='full')[:-2] for reward in reward_list]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(result[i], color_list[i], label=label_list[i])

        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('Training Episode (every 100 episode)')
        plt.title('Reward per Agent')
        plt.savefig("reward_plot.png")
        plt.close()

def plot_total_distance():
    with open('../NUM_AGENT05/0005/distance_output/total_distance_list-NUM_AGENT05-Random_seed0005.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]

        color_list = utils.color_list
        label_list = ['agent' + str(i+1) for i in range(NUMBER_OF_AGENT)]

        total_distance_list = [list(map(float, list1[i])) for i in range(NUMBER_OF_AGENT)]
        print(len(total_distance_list[0]))

        result = [np.convolve(dist, np.ones((2,)) / 2, mode='full')[:-2] for dist in total_distance_list]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(result[i], color_list[i], label=label_list[i])

        plt.legend()
        plt.ylabel('Total Step Sizes')
        plt.xlabel('Training Episode (every 100 episode)')
        plt.title('Total Distance per Agent')
        plt.savefig("total_distance_plot.png")
        plt.close()

def plot_qmax():
    with open('./greedy_output/q_max.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = [row for row in readCSV]

        color_list = utils.color_list
        label_list = ['agent' + str(i+1) for i in range(NUMBER_OF_AGENT)]

        qmax_list = [list(map(float, list1[i])) for i in range(NUMBER_OF_AGENT)]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(qmax_list[i], color_list[i], label=label_list[i] + '-qmax')

        plt.ylabel('Q-max')
        plt.xlabel('Training Steps')
        plt.title('Q-max per Agent')
        plt.legend()
        plt.savefig("qmax_plot.png")
        plt.close()

NUMBER_OF_AGENT = 5
plot_reward(NUMBER_OF_AGENT)
plot_total_distance()
# plot_qmax()