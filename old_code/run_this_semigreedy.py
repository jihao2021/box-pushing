import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
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
#max_epLength = 500
### i need to change max_epLength to make it smaller otherwise simulation will be too slow.
max_epLength = 200

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

                """Swap observation"""
                observation = observation_

                episode_steps += 1

                RL.step += 1
                for i in range(NUMBER_OF_AGENT):
                    Boxer_list[i].step += 1

                #Change this as it makes more sense
                if done["Boxer1"] == True:
                    if env.reach_goal == True:
                        reach_goal_history.append(1)
                    else:
                        reach_goal_history.append(0)

                """break while loop when end of this episode or when episode is too many"""
                if done['Boxer1'] or episode_steps > max_epLength:
                    break
            print("episode is :", episode)

        print('game is over')
        success_rate = reach_goal_history.count(1) / number_of_test_step
        success_rate_list.append(success_rate)
        print('success rate is:', success_rate)

        #output_path = "./semigreedy_output"
        #if not os.path.exists(output_path):
        #    os.makedirs(output_path)

        output_path_semigreedy = "./NUM_AGENT" + str(NUMBER_OF_AGENT).zfill(2) + "/" + str(random_seed).zfill(4) +'/semigreedy_output'
        if not os.path.exists(output_path_semigreedy):
            os.makedirs(output_path_semigreedy)

        output_path_semigreedy_final = output_path_semigreedy + '/' + str(NUMBER_OF_AGENT_TRAINED_MODEL).zfill(2)
        if not os.path.exists(output_path_semigreedy_final):
            os.makedirs(output_path_semigreedy_final)
    '''
    with open(os.path.join(output_path, 'reach_goal_history.csv'), 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerow(success_rate_list)
        #wr.writerow([])
        data_dump.close()
    '''

    with open(os.path.join(output_path_semigreedy_final,
                           'reach_goal_history'+
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
        wr.writerow(success_rate_list)
        #wr.writerow([])
        data_dump.close()

    '''End of run_greedy algorithm'''
########################################################################################

if __name__== "__main__":
    start = time.time()
    #input = 'pycharm'
    input = 'shell'
    loading_model = True
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
        NUMBER_OF_AGENT_TRAINED_MODEL = args.number_agent_trained_model
        random_seed_trained_model = args.random_seed_trained_model
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    elif input == 'pycharm':
        NUMBER_OF_AGENT = 4
        random_seed=5
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

    run_game_semigreedy(number_of_test_step=500, random_percentage=[0.1, 0.3, 0.5])
    end = time.time()
    print('total time is: {} seconds'.format(str(end-start)))
