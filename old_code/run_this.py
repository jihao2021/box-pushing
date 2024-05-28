import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from .pygame_env import CustomEnv
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import csv
import os
import os.path
#from mpl_toolkits.mplot3d import Axes3D
import constant
import time

from DQN_modified_plus_simplified import DeepQNetwork
from Boxer import Boxer


#num_episodes = 5000
num_episodes = 8000
max_epLength = 500

def run_game():
    reward_list = [[] for _ in range(NUMBER_OF_AGENT)]
    RL.step = 0
    for i in range(NUMBER_OF_AGENT):
        Boxer_list[i].step = 0

    total_distance_list = [[] for _ in range(NUMBER_OF_AGENT)]

    for episode in range(1,num_episodes+1):
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

            #env.render()


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
            print("Episode {}, average reward of last 25 episode of Boxer1 is {}".format(episode, np.mean(reward_list[0][-25:])))
            print("epsilon of Boxer1 is :", Boxer_list[0].epsilon)

    print('game is all finished, fight on!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    for i in range(NUMBER_OF_AGENT):
        Boxer_list[i].saver.save(Boxer_list[i].sess, Boxer_list[i].path + '/model-' + str(episode) + RL.Boxer_name[i] + '.ckpt')

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

    #output_path = "./training_output"
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)

    output_path_reward = "./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" + str(random_seed).zfill(4) +'/reward_output'
    if not os.path.exists(output_path_reward):
        os.makedirs(output_path_reward)

    output_path_distance = "./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" + str(random_seed).zfill(4) +'/distance_output'
    if not os.path.exists(output_path_distance):
        os.makedirs(output_path_distance)

    with open(os.path.join(output_path_reward,
                           'reward_history'+
                           '-NUM_AGENT'+
                           str(NUMBER_OF_AGENT).zfill(2)+
                           '-Random_seed'+
                           str(random_seed).zfill(4)+
                           '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            #wr.writerow(resized_list_reward_mean[i])
            wr.writerow(reward_list[i])
        data_dump.close()

    with open(os.path.join(output_path_distance,
                           'total_distance_list'+
                           '-NUM_AGENT'+
                           str(NUMBER_OF_AGENT).zfill(2)+
                           '-Random_seed'+
                           str(random_seed).zfill(4)+
                           '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            #wr.writerow(resized_list_total_distance_mean[i])
            wr.writerow(total_distance_list[i])
        data_dump.close()

'''End of run_game algorithm'''
###############################################################################################3

def run_game_greedy():
    RL.step = 0
    for i in range(NUMBER_OF_AGENT):
        Boxer_list[i].step = 0

    Box_position_list_x =[]
    Box_position_list_y = []

    Boxer_qmax_list = [[] for _ in range(NUMBER_OF_AGENT)]

    for episode in range(1):
        observation = env.reset()
        Box1 = env.object_boxes[0]

        episode_steps = 0

        episode_reward = []
        for i in range(NUMBER_OF_AGENT):
            episode_reward.append(0)

        while True:
            """fresh env"""
            env.render()
            print("Box position vector is:", Box1.position_vector)

            '''Get q_max value for each boxer'''

            Boxer_qmax_value_list = []
            for i in range(NUMBER_OF_AGENT):
                Boxer_qmax_value_list.append(Boxer_list[i].get_qmax(observation[RL.Boxer_name[i]]))


            Box_position_list_x.append(Box1.position_vector.x)
            Box_position_list_y.append(Box1.position_vector.y)

            for i in range(NUMBER_OF_AGENT):
                Boxer_qmax_list[i].append(Boxer_qmax_value_list[i])

            """RL choose action based on observation"""
            action = RL.choose_action_greedy(observation, epsilon_greedy=0)

            """Take action and get next observation and reward"""

            observation_, reward, done, _ = env.step(action, RL.Boxer_name)

            for i in range(NUMBER_OF_AGENT):
                episode_reward[i] += reward[RL.Boxer_name[i]]

            """Swap observation"""
            observation = observation_

            episode_steps += 1

            RL.step += 1
            for i in range(NUMBER_OF_AGENT):
                Boxer_list[i].step += 1

            """break while loop when end of this episode or when episode is too many"""
            if done['Boxer1'] or episode_steps > max_epLength:
                break
        print("episode is :", episode)

    print('game is over')

    output_path = "./greedy_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'q_max.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            wr.writerow(Boxer_qmax_list[i])
        data_dump.close()

    '''End of run_greedy algorithm'''
########################################################################################

'''Semi-greedy algorithm start here'''
def run_game_semigreedy(number_of_test_step,random_percentage):
    success_rate_list=[]
    for id, random_number in enumerate(random_percentage):
        print("Round:", id+1)
        RL.step = 0
        for i in range(NUMBER_OF_AGENT):
            Boxer_list[i].step = 0

        reach_goal_history = []
        for episode in range(1,number_of_test_step+1):
            observation = env.reset()

            episode_steps = 0

            while True:
                """fresh env"""
                env.render()
                action = RL.choose_action_greedy(observation, epsilon_greedy=random_number)

                """Take action and get next observation and reward"""

                observation_, reward, done, _ = env.step(action, RL.Boxer_name)

                if done["Boxer1"] == True:
                    if env.reach_goal == True:
                        reach_goal_history.append(1)
                    else:
                        reach_goal_history.append(0)

                """Swap observation"""
                observation = observation_

                episode_steps += 1

                RL.step += 1
                for i in range(NUMBER_OF_AGENT):
                    Boxer_list[i].step += 1

                """break while loop when end of this episode or when episode is too many"""
                if done['Boxer1'] or episode_steps > max_epLength:
                    break
            print("episode is :", episode)

        print('game is over')
        success_rate = reach_goal_history.count(1) / number_of_test_step
        success_rate_list.append(success_rate)
        print('success rate is:', success_rate)

        output_path = "./semigreedy_output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    with open(os.path.join(output_path, 'reach_goal_history.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerow(success_rate_list)
        #wr.writerow([])
        data_dump.close()
    '''End of run_greedy algorithm'''
########################################################################################

if __name__== "__main__":
    start = time.time()
    loading_model = False
    '''

    To be consistent with run_this_semigreedy model, i added argument parser 
    number_agent_trained_model and random_seed_trained_model, 
    program run_this actually do not need this two and can be without passing these arguments


    '''
    #input = 'pycharm'
    input = 'shell'
    if input == 'shell':
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
        random_seed=args.random_seed
        NUMBER_OF_AGNET_TRAINED_MODEL = args.number_agent_trained_model
        random_seed_trained_model = args.random_seed_trained_model
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    elif input == 'pycharm':
        NUMBER_OF_AGENT = 4
        random_seed=5
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

    #NUMBER_OF_AGENT = 4
    print("number of agent is:", NUMBER_OF_AGENT)
    env = CustomEnv(random_initialization=False, n_agents=NUMBER_OF_AGENT)
    
    RL = DeepQNetwork(n_agents=NUMBER_OF_AGENT,
                      n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      load_model=loading_model,
                      #path="./dqn",
                      path="./NUM_AGENT" +str(NUMBER_OF_AGENT).zfill(2) +"/" +
                           str(random_seed).zfill(4) +'/saved_model',
                      output_graph=True,
                      random_seed=random_seed,
                      n_agents_trained_model=NUMBER_OF_AGNET_TRAINED_MODEL,
                      random_seed_trained_model=random_seed_trained_model,
                      number_episode=num_episodes
                      )
    Boxer_list =[]
    for i in range(NUMBER_OF_AGENT):
        Boxer_list.append(RL.Boxer[i])

    run_game()
    end = time.time()
    #print('total time of this simulation case is:  ', end - start)
    print('total time is: {} seconds'.format(str(end-start)))

    #run_game_semigreedy(number_of_test_step=1000, random_percentage=[0.1, 0.3, 0.5])

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
