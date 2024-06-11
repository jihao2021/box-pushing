#!/usr/bin/env python
# coding: utf-8

# In[48]:


import os

# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
from scripts.pygame_env import CustomEnv
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv
import os
import os.path
# from mpl_toolkits.mplot3d import Axes3D
from scripts import constant
import time

from scripts.DQN_modified_plus_simplified import DeepQNetwork
from scripts.Boxer import Boxer

# num_episodes = 5000
num_episodes = 5000
max_epLength = 500

NUMBER_OF_AGENT = 10
NUMBER_OF_AGNET_TRAINED_MODEL = 0


# In[44]:


def run_game():
    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    RL.step = 0
    for i in range(NUMBER_OF_AGENT):
        Boxer_list[i].step = 0

    total_distance_list = [[] for _ in range(NUMBER_OF_AGENT)]

    for episode in range(1, num_episodes + 1):
        observation = env.reset()
        pusher_list = []
        for i in range(NUMBER_OF_AGENT):
            pusher_list.append(env.object_pushers[i])

        episode_steps = 0
        episode_reward_list = []

        for i in range(NUMBER_OF_AGENT):
            episode_reward_list.append(0)

        while True:
            """fresh env"""

            # env.render()

            """RL choose action based on observation"""

            action = RL.choose_action(observation)

            """Take action and get next observation and reward"""

            observation_, reward, done, _ = env.step(action, RL.Boxer_name)

            for i in range(NUMBER_OF_AGENT):
                episode_reward_list[i] += reward[RL.Boxer_name[i]]

            RL.store_transition(observation, action, reward, observation_, done)

            for i in range(NUMBER_OF_AGENT):
                if (Boxer_list[i].step > Boxer_list[i].pre_train_steps) and (Boxer_list[i].step % 5 == 0):
                    Boxer_list[i].learn()

            """Swap observation"""
            observation = observation_

            episode_steps += 1

            RL.step += 1
            for i in range(NUMBER_OF_AGENT):
                Boxer_list[i].step += 1

            """break while loop when end of this episode or when episode is too many"""
            if done['Boxer1'] or episode_steps > max_epLength:

                # if done['Boxer1'] or pusher1.distance_travelled > 60000 or pusher2.distance_travelled >60000 or pusher3.distance_travelled > 60000:

                '''Save total distance and reward every episode'''
                for i in range(NUMBER_OF_AGENT):
                    total_distance_list[i].append(pusher_list[i].distance_travelled)

                for i in range(NUMBER_OF_AGENT):
                    reward_list[i].append(episode_reward_list[i])
                break

        print("episode is :", episode)
        print("epsilon of Boxer1 is :", Boxer_list[0].epsilon)

        if episode > 0 and episode % 25 == 0:
            print("Episode {}, average reward of last 25 episode of Boxer1 is {}".format(episode,
                                                                                         np.mean(reward_list[0][-25:])))
            print("epsilon of Boxer1 is :", Boxer_list[0].epsilon)

    print('game is all finished, fight on!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    RL.save_models()

    '''
    resized_list_reward = []
    resized_list_reward_mean = []
    for i in range(NUMBER_OF_AGENT):
        resized_list_reward.append(np.resize(np.array(reward_list[i]), [len(reward_list[i]) // 100, 100]))
        resized_list_reward_mean.append(np.average(resized_list_reward[i], 1))

    resized_list_total_distance = []
    resized_list_total_distance_mean = []
    for i in range(NUMBER_OF_AGENT):
        resized_list_total_distance.append(np.resize(np.array(total_distance_list[i]),[len(total_distance_list[i]) // 100, 100] ))
        resized_list_total_distance_mean.append(np.average(resized_list_total_distance[i],1))
    '''

    # output_path = "./training_output"
    # if not os.path.exists(output_path):
    #    os.makedirs(output_path)

    output_path_reward = "./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" + str(random_seed).zfill(
        4) + '/reward_output'
    if not os.path.exists(output_path_reward):
        os.makedirs(output_path_reward)

    output_path_distance = "./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" + str(random_seed).zfill(
        4) + '/distance_output'
    if not os.path.exists(output_path_distance):
        os.makedirs(output_path_distance)

    with open(os.path.join(output_path_reward,
                           'reward_history' +
                           '-NUM_AGENT' +
                           str(NUMBER_OF_AGENT).zfill(2) +
                           '-Random_seed' +
                           str(random_seed).zfill(4) +
                           '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            # wr.writerow(resized_list_reward_mean[i])
            wr.writerow(reward_list[i])
        data_dump.close()

    with open(os.path.join(output_path_distance,
                           'total_distance_list' +
                           '-NUM_AGENT' +
                           str(NUMBER_OF_AGENT).zfill(2) +
                           '-Random_seed' +
                           str(random_seed).zfill(4) +
                           '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            # wr.writerow(resized_list_total_distance_mean[i])
            wr.writerow(total_distance_list[i])
        data_dump.close()


'''End of run_game algorithm'''

# In[50]:


if __name__ == "__main__":
    start = time.time()
    loading_model = False
    '''

    To be consistent with run_this_semigreedy model, i added argument parser 
    number_agent_trained_model and random_seed_trained_model, 
    program run_this actually do not need this two and can be without passing these arguments


    '''
    # input = 'pycharm'
    input_ = 'pycharm'
    if input_ == 'shell':
        parser = argparse.ArgumentParser()
        parser.add_argument('-number_agent', type=int, default=3,
                            help='choose number of agents')
        parser.add_argument('-random_seed', type=int, default=1,
                            help='choose the random seed number')
        parser.add_argument('-number_agent_trained_model', type=int, default=1,
                            help='choose the number of agnet of trained model')
        parser.add_argument('-random_seed_trained_model', type=int, default=1,
                            help='choose the random seed number associated with trained models')
        args = parser.parse_args()
        print('=' * 50)
        print(args)
        print('=' * 50)
        NUMBER_OF_AGENT = args.number_agent
        random_seed = args.random_seed
        NUMBER_OF_AGNET_TRAINED_MODEL = args.number_agent_trained_model
        random_seed_trained_model = args.random_seed_trained_model
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    elif input_ == 'pycharm':
        NUMBER_OF_AGENT = 5
        random_seed = 5
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

    # NUMBER_OF_AGENT = 4
    print("number of agent is:", NUMBER_OF_AGENT)
    env = CustomEnv(random_initialization=False, n_agents=NUMBER_OF_AGENT)

    RL = DeepQNetwork(n_agents=NUMBER_OF_AGENT,
                      n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      load_model=loading_model,
                      # path="./dqn",
                      path="./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" +
                           str(random_seed).zfill(4) + '/saved_model',
                      output_graph=True,
                      random_seed=random_seed,
                      n_agents_trained_model=NUMBER_OF_AGNET_TRAINED_MODEL,
                      random_seed_trained_model=None,
                      number_episode=num_episodes
                      )

    Boxer_list = []
    for i in range(NUMBER_OF_AGENT):
        Boxer_list.append(RL.Boxer[i])

    print("action: ", RL.choose_action(env.reset()))
    print("boxer name: ", RL.Boxer_name)

    run_game()
    end = time.time()


    RL.save_models()
    # print('total time of this simulation case is:  ', end - start)
    print('total time is: {} seconds'.format(str(end - start)))

    # run_game_semigreedy(number_of_test_step=1000, random_percentage=[0.1, 0.3, 0.5])

    '''
    if loading_model:
        option_greedy = False

        if option_greedy == True:
            run_game_greedy()
        else:
            run_game_semigreedy(number_of_test_step=1000, random_percentage=[0.1,0.3,0.5])
            ##remember to change env reset option if using random initialization
    
    else:
        run_game()
    '''

# In[ ]:
