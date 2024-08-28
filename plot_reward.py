import constant
import csv
import numpy as np
import matplotlib.pyplot as plt


def plot_reward(NUMBER_OF_AGENT):
    list1 = []
    result = [[] for _ in range(NUMBER_OF_AGENT)]
    color_list = constant.color_list

    label_list = []
    for i in range(NUMBER_OF_AGENT):
        label_list.append('agent' + str(i + 1))
    reward_list = [[] for _ in range(NUMBER_OF_AGENT+NUMBER_OF_AGENT)]
    result = [[] for _ in range(NUMBER_OF_AGENT)]

    with open('./training_output/reward_output/reward15.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')

        for row in readCSV:
            list1.append(row)
            #print(list1)


        for i in range(NUMBER_OF_AGENT):
            reward_list[i] = list1[i]
            reward_list[i] = list(map(float,reward_list[i]))

        #print(reward_list)

        #for i in range(NUMBER_OF_AGENT):
            #result[i] = np.convolve(reward_list[i], np.ones((2,)) / 2, mode='full')
            #result[i] = result[i][:-2]

    with open('./training_output/reward_output/reward15.csv') as csvfile:
        #print(list1)
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            list1.append(row)
        #print(np.shape(list1))


        for i in range(NUMBER_OF_AGENT):
            reward_list[i+NUMBER_OF_AGENT] = list1[i+NUMBER_OF_AGENT]
            reward_list[i+NUMBER_OF_AGENT] = list(map(float, reward_list[i+NUMBER_OF_AGENT]))

        for i in range(NUMBER_OF_AGENT):
            reward_list[i] = [(x+y)/2 for x,y in zip(reward_list[i],reward_list[i+NUMBER_OF_AGENT])]
            #pass
            #reward_list[i + NUMBER_OF_AGENT] = list(map(float, reward_list[i + NUMBER_OF_AGENT]))

        for i in range(NUMBER_OF_AGENT):
            result[i] = np.convolve(reward_list[i], np.ones((2,)) / 2, mode='full')
            result[i] = result[i][:-2]

        for i in range(NUMBER_OF_AGENT):
            plt.plot(result[i], color_list[i],label=label_list[i])

        plt.legend()
        plt.ylabel('Reward')
        plt.xlabel('training episode(every 100 episode)')
        plt.show()

NUMBER_OF_AGENT = 15
plot_reward(NUMBER_OF_AGENT)
