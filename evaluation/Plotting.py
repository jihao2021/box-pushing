import csv
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
from scripts import constant

def plot_reward(NUMBER_OF_AGENT):
    with open('./training_output/reward_history.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)

        color_list = constant.color_list

        label_list = []
        for i in range(NUMBER_OF_AGENT):
            label_list.append('agent' + str(i+1))

        reward_list = [[] for _ in range(NUMBER_OF_AGENT)]

        for i in range(NUMBER_OF_AGENT):
            reward_list[i] = list1[i]
            reward_list[i] = list(map(float,reward_list[i]))

        result = [[] for _ in range(NUMBER_OF_AGENT)]
        for i in range(NUMBER_OF_AGENT):
            result[i] = np.convolve(reward_list[i], np.ones((2,)) / 2, mode='full')
            result[i] = result[i][:-2]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(result[i], color_list[i],label=label_list[i])
        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('training episode(every 100 episode)')
        plt.show()

def plot_total_distance():
    with open('./training_output/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)

        color_list = constant.color_list

        label_list = []
        for i in range(NUMBER_OF_AGENT):
            label_list.append('agent' + str(i + 1))

        total_distance_list = [[] for _ in range(NUMBER_OF_AGENT)]

        for i in range(NUMBER_OF_AGENT):
            total_distance_list[i] = list1[i]
            total_distance_list[i] = list(map(float,total_distance_list[i]))

        print(len(total_distance_list[0]))

        result = [[] for _ in range(NUMBER_OF_AGENT)]
        for i in range(NUMBER_OF_AGENT):
            result[i] = np.convolve(total_distance_list[i], np.ones((2,)) / 2, mode='full')
            result[i] = result[i][:-2]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(result[i], color_list[i],label=label_list[i])
            #plt.plot(total_distance_list[i], color_list[i], label=label_list[i])


        # total_distance_list_all = [sum(x) for x in zip(*total_distance_list)]
        #
        # result_total = np.convolve(total_distance_list_all, np.ones((2,)) / 2, mode='full')
        # result_total = result_total[: -2]
        # plt.plot(result_total, 'k-', label='total_distance_all')

        plt.legend()

        plt.ylabel('Total Step Sizes')
        plt.xlabel('training episode(every 100 episode)')
        plt.show()

def plot_qmax():
    with open('./greedy_output/q_max.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        color_list = constant.color_list

        label_list = []
        for i in range(NUMBER_OF_AGENT):
            label_list.append('agent' + str(i + 1))

        qmax_list = [[] for _ in range(NUMBER_OF_AGENT)]
        for i in range(NUMBER_OF_AGENT):
            qmax_list[i] = list1[i]
            qmax_list[i] = list(map(float, qmax_list[i]))

        for i in range(NUMBER_OF_AGENT):
            plt.plot(qmax_list[i], color_list[i],label=label_list[i] + '-qmax')
        plt.ylabel('qmax')
        plt.xlabel('Training Steps')
        plt.legend()
        plt.show()

NUMBER_OF_AGENT = 4
plot_reward(NUMBER_OF_AGENT)
plot_total_distance()
#plot_qmax()






