import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_reward():
    with open('./training_result_archive/-1-1-1-0-0-0-5000/seed1/reward_history.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        reward_list1 = list1[0][:]
        reward_list2 = list1[1][:]
        reward_list3 = list1[2][:]

        reward_list11 = list(map(float, reward_list1))
        reward_list21 = list(map(float, reward_list2))
        reward_list31 = list(map(float, reward_list3))

    with open('./training_result_archive/-1-1-1-0-0-0-5000/seed2/reward_history.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        reward_list1 = list1[0][:]
        reward_list2 = list1[1][:]
        reward_list3 = list1[2][:]

        reward_list12 = list(map(float, reward_list1))
        reward_list22 = list(map(float, reward_list2))
        reward_list32 = list(map(float, reward_list3))

    with open('./training_result_archive/-1-1-1-0-0-0-5000/seed3/reward_history.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        reward_list1 = list1[0][:]
        reward_list2 = list1[1][:]
        reward_list3 = list1[2][:]

        reward_list13 = list(map(float, reward_list1))
        reward_list23 = list(map(float, reward_list2))
        reward_list33 = list(map(float, reward_list3))

    # with open('./training_result_archive/-1-1-0-0-0-0-5000/seed4/reward_history.csv') as csvfile:
    #     readCSV = csv.reader(csvfile, delimiter=',')
    #     list1 = []
    #     for row in readCSV:
    #         list1.append(row)
    #     reward_list1 = list1[0][:]
    #     reward_list2 = list1[1][:]
    #     reward_list3 = list1[2][:]
    #
    #     reward_list14 = list(map(float, reward_list1))
    #     reward_list24 = list(map(float, reward_list2))
    #     reward_list34 = list(map(float, reward_list3))
    #
    # with open('./training_result_archive/-1-1-0-0-0-0-5000/seed5/reward_history.csv') as csvfile:
    #     readCSV = csv.reader(csvfile, delimiter=',')
    #     list1 = []
    #     for row in readCSV:
    #         list1.append(row)
    #     reward_list1 = list1[0][:]
    #     reward_list2 = list1[1][:]
    #     reward_list3 = list1[2][:]
    #
    #     reward_list15 = list(map(float, reward_list1))
    #     reward_list25 = list(map(float, reward_list2))
    #     reward_list35 = list(map(float, reward_list3))

        # reward_list1 = np.add(reward_list11, reward_list12, reward_list13, reward_list14, reward_list15).tolist()
        # reward_list2 = np.add(reward_list21, reward_list22, reward_list23, reward_list24, reward_list25).tolist()
        # reward_list3 = np.add(reward_list31, reward_list32, reward_list33, reward_list34, reward_list35).tolist()

        # reward_list1 = [sum(x)/5 for x in zip(reward_list11, reward_list12, reward_list13, reward_list14, reward_list15)]
        # reward_list2 = [sum(x)/5 for x in zip(reward_list21, reward_list22, reward_list23, reward_list24, reward_list25)]
        # reward_list3 = [sum(x)/5 for x in zip(reward_list31, reward_list32, reward_list33, reward_list34, reward_list35)]
        #
        reward_list1 = [sum(x) / 3 for x in
                        zip(reward_list11, reward_list12, reward_list13, )]
        reward_list2 = [sum(x) / 3 for x in
                        zip(reward_list21, reward_list22, reward_list23, )]
        reward_list3 = [sum(x) / 3 for x in
                        zip(reward_list31, reward_list32, reward_list33, )]

        plt.plot(reward_list1, 'r--')
        plt.plot(reward_list2,'b-.')
        plt.plot(reward_list3,'g:')
        # result1 = np.convolve(reward_list1, np.ones((2,)) / 2, mode='full')
        # result1 = result1[: -2]
        # # plt.plot(resized_list_mean_Boxer1,'k-', label='reward1')
        # plt.plot(result1, 'r--', label='reward1')
        #
        # result2 = np.convolve(reward_list2, np.ones((2,)) / 2, mode='full')
        # result2 = result2[: -2]
        # plt.plot(result2, 'b-.', label='reward2')
        # result3 = np.convolve(reward_list3, np.ones((2,)) / 2, mode='full')
        # result3 = result3[: -2]
        # plt.plot(result3, 'g:', label='reward3')

        plt.ylabel('Reward')
        plt.xlabel('training episode(every 100 episode)')
        plt.show()

def plot_total_distance():
    with open('./training_result_archive/-1-1-0-0-0-0-5000/seed1/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        total_distance_list1 = list1[0][:]
        total_distance_list2 = list1[1][:]
        total_distance_list3 = list1[2][:]

        total_distance_list11 = list(map(float, total_distance_list1))
        total_distance_list21 = list(map(float, total_distance_list2))
        total_distance_list31 = list(map(float, total_distance_list3))

    with open('./training_result_archive/-1-1-0-0-0-0-5000/seed2/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        total_distance_list1 = list1[0][:]
        total_distance_list2 = list1[1][:]
        total_distance_list3 = list1[2][:]

        total_distance_list12 = list(map(float, total_distance_list1))
        total_distance_list22 = list(map(float, total_distance_list2))
        total_distance_list32 = list(map(float, total_distance_list3))

    with open('./training_result_archive/-1-1-0-0-0-0-5000/seed3/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        total_distance_list1 = list1[0][:]
        total_distance_list2 = list1[1][:]
        total_distance_list3 = list1[2][:]

        total_distance_list13 = list(map(float, total_distance_list1))
        total_distance_list23 = list(map(float, total_distance_list2))
        total_distance_list33 = list(map(float, total_distance_list3))

    with open('./training_result_archive/-1-1-0-0-0-0-5000/seed4/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        total_distance_list1 = list1[0][:]
        total_distance_list2 = list1[1][:]
        total_distance_list3 = list1[2][:]

        total_distance_list14 = list(map(float, total_distance_list1))
        total_distance_list24 = list(map(float, total_distance_list2))
        total_distance_list34 = list(map(float, total_distance_list3))

    with open('./training_result_archive/-1-1-0-0-0-0-5000/seed5/total_distance_list.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        total_distance_list1 = list1[0][:]
        total_distance_list2 = list1[1][:]
        total_distance_list3 = list1[2][:]

        total_distance_list15 = list(map(float, total_distance_list1))
        total_distance_list25 = list(map(float, total_distance_list2))
        total_distance_list35 = list(map(float, total_distance_list3))

        distance_list1 = [sum(x) / 5 for x in
                        zip(total_distance_list11, total_distance_list12, total_distance_list13, total_distance_list14, total_distance_list15)]
        distance_list2 = [sum(x) / 5 for x in
                          zip(total_distance_list21, total_distance_list22, total_distance_list23,
                              total_distance_list24, total_distance_list25)]
        distance_list3 = [sum(x) / 5 for x in
                          zip(total_distance_list31, total_distance_list32, total_distance_list33,
                              total_distance_list34, total_distance_list35)]

        plt.plot(distance_list1, 'r--')
        plt.plot(distance_list2, 'b-.')
        plt.plot(distance_list3, 'g:')
        # result1 = np.convolve(total_distance_list1, np.ones((2,)) / 2, mode='full')
        # result1 = result1[: -2]
        # # plt.plot(resized_list_mean_Boxer1,'k-', label='reward1')
        # plt.plot(result1, 'r--', label='reward1')
        # result2 = np.convolve(total_distance_list2, np.ones((2,)) / 2, mode='full')
        # result2 = result2[: -2]
        # plt.plot(result2, 'b-.', label='reward2')
        # result3 = np.convolve(total_distance_list3, np.ones((2,)) / 2, mode='full')
        # result3 = result3[: -2]
        # plt.plot(result3, 'g:', label='reward3')

        total_distance_list1list2list3 = [a + b + c for a, b, c in
                                          zip(distance_list1,distance_list2, distance_list3)]
        # result_total = np.convolve(total_distance_list1list2list3, np.ones((2,)) / 2, mode='full')
        # result_total = result_total[: -2]
        plt.plot(total_distance_list1list2list3, 'k-')

        plt.ylabel('Total Distance')
        plt.xlabel('training episode(every 100 episode)')
        plt.show()

def plot_qmax():
    with open('./training_result_archive/-1-0-0-0-0-0-5000/seed4/q_max.csv') as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        list1 = []
        for row in readCSV:
            list1.append(row)
        qmax_list1 = list1[0][:]
        qmax_list2 = list1[1][:]
        qmax_list3 = list1[2][:]

        qmax_list1 = list(map(float, qmax_list1))
        qmax_list2 = list(map(float, qmax_list2))
        qmax_list3 = list(map(float, qmax_list3))

        plt.plot(qmax_list1, 'r--', label='reward1')
        plt.plot(qmax_list2, 'b-.', label='reward2')
        plt.plot(qmax_list3, 'g:', label='reward3')
        plt.ylabel('qmax')
        plt.xlabel('Training Steps')
        plt.show()



plot_reward()
plot_total_distance()