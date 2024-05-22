#import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"
from pygame_env import CustomEnv
import argparse
import pygame as pg
from DQN_modified_plus_simplified import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Boxer import Boxer
import csv
import os
import os.path
#from mpl_toolkits.mplot3d import Axes3D
import constant
import time

num_episodes = 8000
max_epLength = 200

def run_game_greedy():
    RL.step = 0
    for i in range(NUMBER_OF_AGENT):
        Boxer_list[i].step = 0

    Box_position_list_x =[]
    Box_position_list_y = []
    Box_heading_list = []

    Boxer_qmax_list = [[] for _ in range(NUMBER_OF_AGENT)]



    

    for episode in range(1):
        observation = env.reset()
        Box1 = env.object_boxes[0]

        episode_steps = 0

        episode_reward = []

        reach_goal_history = []
        reach_goal_trajectory = []
        rotation_list = []
        x_list = []
        y_list = []


        for i in range(NUMBER_OF_AGENT):
            episode_reward.append(0)

        while True:
            """fresh env"""
            env.render()
            #print("Box position vector is:", Box1.position_vector)

            '''Get q_max value for each boxer'''

            Boxer_qmax_value_list = []

            action_list=[]
            for i in range(NUMBER_OF_AGENT):
                Boxer_qmax_value_list.append(Boxer_list[i].get_qmax(observation[RL.Boxer_name[i]]))

            Box_position_list_x.append(Box1.position_vector.x)
            Box_position_list_y.append(Box1.position_vector.y)
            Box_heading_list.append(Box1.current_heading)

            for i in range(NUMBER_OF_AGENT):
                Boxer_qmax_list[i].append(Boxer_qmax_value_list[i])

            """RL choose action based on observation"""
            action = RL.choose_action_greedy(observation, epsilon_greedy=0)


            #print(action)
            for i in range(NUMBER_OF_AGENT):
                action_list.append(action[RL.Boxer_name[i]])
            #print(action_list)

            reach_goal_trajectory.append(action_list)

            Number_of_0 = action_list.count(0)  #region 1 one-hit
            Number_of_1 = action_list.count(1)  #region 2 two-hit
            Number_of_2 = action_list.count(2)  #region 3 three-hit
            Number_of_3 = action_list.count(3)  #region 4 four-hit
            Number_of_4 = action_list.count(4)  #region 5 five-hit
            Number_of_5 = action_list.count(5)  #region 6 six-hit
            Sum = Number_of_0 + Number_of_1 + Number_of_2 + Number_of_3 + Number_of_4 + Number_of_5
            rotation = Number_of_0 + Number_of_2 - Number_of_1 - Number_of_3
            #change_x = (self.three_hit + self.four_hit - self.one_hit - self.two_hit) 
            #change_y = (self.five_hit - self.six_hit) 
            change_x = Number_of_2 + Number_of_3 - Number_of_0 - Number_of_1
            change_y = Number_of_4 - Number_of_5
            #print(rotation)
            rotation_list.append(rotation)
            x_list.append(change_x)
            y_list.append(change_y)


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

            if done['Boxer1'] or episode_steps > max_epLength:
                #print('yes')
                #print(done['Boxer1'])
                if env.reach_goal == True:
                    reach_goal_history.append(1)
                else:

                    reach_goal_history.append(0)

            """break while loop when end of this episode or when episode is too many"""
            if done['Boxer1'] or episode_steps > max_epLength:
                #print('yes')
                break
        print("episode is :", episode)

    

    output_path_screenshot = "./NUM_AGENT" + str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2) + "/" + str(random_seed_trained_model).zfill(4) + '/screenshot'
    if not os.path.exists(output_path_screenshot):
        os.makedirs(output_path_screenshot)
        
    pg.image.save(env.screen, output_path_screenshot + "/screenshot" + str(random_seed_trained_model).zfill(4) + ".jpg")
    print('game is over')

    
    output_path_greedy_output = "./NUM_AGENT" + str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2) + "/" + str(random_seed_trained_model).zfill(4) + '/greedy_output'
    if not os.path.exists(output_path_greedy_output):
        os.makedirs(output_path_greedy_output)

    with open(os.path.join(output_path_greedy_output, 
                           'check_success'+
                           '-NUM_AGENT'+
                           str(NUMBER_OF_AGENT).zfill(2)+
                           '-Random_seed'+
                           str(random_seed).zfill(4)+
                           '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                           str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)+
                           '-Random_seed_trained_model'+
                           str(random_seed_trained_model).zfill(4)+
                           '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerow(reach_goal_history)
        #wr.writerow([])
        data_dump.close()

    with open(os.path.join(output_path_greedy_output, 
                               'check_trajectory'+
                               '-NUM_AGENT'+
                               str(NUMBER_OF_AGENT).zfill(2)+
                               '-Random_seed'+
                               str(random_seed).zfill(4)+
                               '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                               str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)+
                               '-Random_seed_trained_model'+
                               str(random_seed_trained_model).zfill(4)+
                               '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(len(reach_goal_trajectory)):
            wr.writerow(reach_goal_trajectory[i])
        #wr.writerow([])
        data_dump.close()

    with open(os.path.join(output_path_greedy_output, 
                               'check_position_heading'+
                               '-NUM_AGENT'+
                               str(NUMBER_OF_AGENT).zfill(2)+
                               '-Random_seed'+
                               str(random_seed).zfill(4)+
                               '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                               str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)+
                               '-Random_seed_trained_model'+
                               str(random_seed_trained_model).zfill(4)+
                               '.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(len(Box_position_list_x)):
            total = []
            total.append(rotation_list[i])
            total.append(Box_position_list_x[i])
            total.append(Box_position_list_y[i])
            total.append(Box_heading_list[i])
            total.append(x_list[i])
            total.append(y_list[i])
            wr.writerow(total)
        #wr.writerow([])
        data_dump.close()





    '''
    with open(os.path.join(output_path, 'q_max.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        for i in range(NUMBER_OF_AGENT):
            wr.writerow(Boxer_qmax_list[i])
        data_dump.close()
    '''

    '''End of run_greedy algorithm'''
########################################################################################

if __name__== "__main__":
    start = time.time()
    loading_model = True
    '''
    random_seed is dumpy input as it is greedy algorithm, no random is involved
    number_agent_trained_model and random_seed_trained_model is necessary for run_game_greedy
    '''
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
    NUMBER_OF_AGENT_TRAINED_MODEL = args.number_agent_trained_model
    random_seed_trained_model = args.random_seed_trained_model
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    
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
                      n_agents_trained_model=NUMBER_OF_AGENT_TRAINED_MODEL,
                      random_seed_trained_model=random_seed_trained_model,
                      number_episode=num_episodes
                      )
    Boxer_list =[]
    for i in range(NUMBER_OF_AGENT):
        Boxer_list.append(RL.Boxer[i])
    run_game_greedy()
    end = time.time()
    print('total time is: {} seconds'.format(str(end-start)))

