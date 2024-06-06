from pygame_env import CustomEnv
import argparse
import pygame as pg
import matplotlib.pyplot as plt
import numpy as np
from Boxer import Boxer
import csv
import os
import os.path
import constant
import time
import pygame
import math
import numpy as np
from Box import Box
import constant
from target import target
from obstacle import obstacle
from Box_checker import box_checker
from setup_background import set_up_background_new_new, set_up_background_new_new_white
import random
from Boxer import Boxer
import gym
from gym import spaces
from gym.utils import seeding
from pusher import Pusher
import pandas as pd
import collections



parser = argparse.ArgumentParser()
parser.add_argument('-number_agent', type=int, default=3,
                help='choose number of agents')
parser.add_argument('-number_rotates', type=int, default=1,
                help='choose rotation increment')
args = parser.parse_args()
print('=' * 50)
print(args)
print('=' * 50)
NUMBER_OF_AGENT = args.number_agent
Number_agent = args.number_agent
Number_of_rotates = args.number_rotates
seed_range = 100

random.seed(constant.random_seed)


N_DISCRETE_ACTIONS = 6


class CustomEnv(gym.Env):
    def __init__(self, random_initialization, n_agents):
        super(CustomEnv, self).__init__()

        '''Define action and observation space, They must be gym.space objects'''
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.random_initialization = random_initialization
        self.n_agents = n_agents

        self.Boxer_name_env = []
        for i in range(self.n_agents):
            self.Boxer_name_env.append("Boxer" + str(i + 1))

        high = 1
        self.observation_space = spaces.Box(-high,high,shape=(9,),dtype=np.float32)

        '''Initialize game screen and clock'''
        pygame.init()

        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Box-pushing")

        '''Setup game screen, size'''
        self.size = (constant.framewidth, constant.frameheight)
        self.screen = pygame.display.set_mode(self.size)

        """
        Define collision, reach_goal, done to be False
        """
        self.collision = False
        self.reach_goal = False
        self.done = False

        self.reward_rotation = True

        self.object_boxes = []
        self.object_obstacles = []
        self.object_targets = []
        self.object_patches = []
        self.object_pushers = []

        '''Initialize box_checkers list'''
        self.object_box_checkers = []

        '''Define weights for reward original w1=0.6, w2=0.1 w3=0.1 w4=0.2'''
        #self.w1 = 0.6
        self.w1= 0.6 * 1.0
        #self.w1 = 0.6 * 0.5


        self.w2 = 0.1
        self.w3 = 0.1
        self.w4 = 0.2
        self.w5 = 0
        self.w6 = 0
        if self.reward_rotation == False:
            '''Sum equal to 0.9'''
            self.w1 = 0.6
            self.w2 = 0.1
            self.w3 = 0.0
            self.w4 = 0.2
            self.w5 = 0
            self.w6 = 0

        """
        Create static objects such as target, obstacle and patches
        """
        self.object_targets = []
        #target1 = target(constant.target_x, constant.target_y, target_radius=20, color=constant.white)
        target1 = target(constant.target_x, constant.target_y, target_radius=20, color=constant.black)
        target1.draw_target(self.screen)
        self.object_targets.append(target1)

        self.object_obstacles = []
        obstacle1 = obstacle(constant.obstacle_x, constant.obstacle_y, obstacle_radius=15, color=constant.red)
        obstacle1.draw_obstacle(self.screen)
        self.object_obstacles.append(obstacle1)

        self.object_patches = []
        patch_list = set_up_background_new_new_white(constant.framewidth, constant.frameheight)
        #patch_list = set_up_background_new(constant.framewidth, constant.frameheight)
        self.object_patches = patch_list

        '''
        for patch in patch_list:
            patch.draw_patch(self.screen)
        '''

        """reset the environment"""
        _ = self.reset()

    def reset(self):
        '''reset boxes and pushers to void'''
        self.object_boxes = []
        self.object_pushers = []

        #self.screen.fill(constant.black)
        self.screen.fill(constant.white)

        self.Pusher_name = []
        for i in range(self.n_agents):
            self.Pusher_name.append("Pusher" + str(i + 1))

        '''
        This is independent variable, whether box will deployed randomly or with fixed position
        '''

        if self.random_initialization:
            box1 = Box(initial_x=random.randint(60,181), initial_y=random.randint(120,361), width=constant.box_width, height=constant.box_height)
        else:
            box1 = Box(initial_x=120, initial_y=240, width=constant.box_width, height=constant.box_height)

        #box1.draw_polygon(self.screen)
        #box1.draw_closed_lines(self.screen)
        self.object_boxes.append(box1)
        
        '''Add Pusher and append to object_pusher list'''

        for i in range(self.n_agents):
            self.object_pushers.append(Pusher(size=10, name=self.Pusher_name[i]))

        #for i in range(self.n_agents):
            #self.object_pushers[i].draw_polygon(self.screen)

        '''Add box sensor box_checker'''
        self.add_box_checker()
        state = box1.check_state(self, self.Boxer_name_env)

        '''Initialize goal, collision and done state'''
        self.reach_goal = False
        self.collision = False
        self.done = False
        return state

    def add_box_checker(self):
        '''initialize box checkers'''
        self.object_box_checkers = []

        '''left part of the box checker'''
        x_offset = -constant.box_width / 2
        y_offset = constant.box_height / 2
        for _ in range(9):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            y_offset -= constant.box_height / 8
        x_offset = constant.box_width / 2
        y_offset = constant.box_height / 2

        '''right part of the box checker'''
        for _ in range(9):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            y_offset -= constant.box_height / 8
        y_offset = constant.box_height / 2
        x_offset = - constant.box_width / 4

        '''top part of the box checker'''
        for _ in range(3):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            x_offset += constant.box_width / 4
        y_offset = - constant.box_height / 2
        x_offset = - constant.box_width / 4

        '''bottom part of the box checker'''
        for _ in range(3):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            x_offset += constant.box_width / 4

    def render(self):
        '''Get box and pushers ready'''
        box1 = self.object_boxes[0]

        '''Get target and obstacle, patches ready'''
        target1 = self.object_targets[0]
        obstacle1 = self.object_obstacles[0]
        patch_list = self.object_patches

        #This is the line of code that keeps game from not updating
        #self.screen.fill(constant.black)

        #box1.draw_polygon(self.screen)

        #box1.draw_closed_lines(self.screen)



        #for i in range(self.n_agents):
            #self.object_pushers[i].draw_polygon(self.screen)



        #box1.draw_circle(self.screen)
        #box1.draw_line(self.screen)
        target1.draw_target(self.screen)
        obstacle1.draw_obstacle(self.screen)

        #pygame.draw.line(self.screen, constant.black, [440, 240],
        #                 [460, 240], 5)

        #pygame.draw.line(self.screen, constant.black, [450, 230],
        #                 [450, 250], 5)

        #draw cross marks of target

        pygame.draw.line(self.screen, constant.white, [440, 240],
                         [460, 240], 5)
        pygame.draw.line(self.screen, constant.white, [450, 230],
                         [450, 250], 5)

        # draw left vertical line
        pygame.draw.line(self.screen, constant.black, [constant.frameheight / 20, constant.frameheight / 20],
                         [constant.frameheight / 20, constant.frameheight / 20 * 19], 5)

        #draw first horizontal two lines
        pygame.draw.line(self.screen, constant.black, [constant.frameheight / 20, constant.frameheight / 20],
                         [constant.framewidth / 2, constant.frameheight / 20], 5)
        pygame.draw.line(self.screen, constant.black, [constant.frameheight / 20, constant.frameheight / 20 * 19],
                         [constant.framewidth / 2, constant.frameheight / 20 * 19], 5)

        #draw skew lines
        pygame.draw.line(self.screen, constant.black, [constant.framewidth / 2, constant.frameheight / 20],
                         [constant.framewidth / 4 * 3, constant.frameheight / 4 ], 5)
        pygame.draw.line(self.screen, constant.black, [constant.framewidth / 2, constant.frameheight / 20 * 19],
                         [constant.framewidth / 4 * 3, constant.frameheight / 4 * 3], 5)

        #draw last two horizontal lines
        pygame.draw.line(self.screen, constant.black, [constant.framewidth / 4 * 3, constant.frameheight / 4],
                         [constant.framewidth, constant.frameheight / 4 ], 5)
        pygame.draw.line(self.screen, constant.black, [constant.framewidth / 4 * 3, constant.frameheight / 4 * 3],
                         [constant.framewidth, constant.frameheight / 4 * 3], 5)

        


        

        '''
        for patch in patch_list:
            patch.draw_patch(self.screen)
        '''

        #for box_checker in self.object_box_checkers:
            #box_checker.draw_box_checker(self.screen)

        #time.sleep(0.1)
        pygame.display.update()
        self.clock.tick(constant.FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                sys.exit()

env = CustomEnv(random_initialization=False, n_agents=NUMBER_OF_AGENT)

seed_list = np.arange(1,seed_range+1)
data_frame_list_greedy = [[] for _ in range(len(seed_list))]
random_seed=1
path = './NUM_AGENT' + str(Number_agent).zfill(2) + '/'

for i, item in enumerate(seed_list):
    data_frame_list_greedy[i]=pd.read_csv(path + 
                                    str(item).zfill(4) + 
                                    '/greedy_output/'+
                                    'check_success'+
                                    '-NUM_AGENT'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed'+
                                    str(random_seed).zfill(4)+
                                    '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed_trained_model'+
                                    str(item).zfill(4) +
                                    '.csv',header=None)


df_final_greedy = data_frame_list_greedy[0]
for i in range(len(data_frame_list_greedy)-1):
  df_final_greedy = pd.concat([df_final_greedy, data_frame_list_greedy[i+1]], ignore_index=True)
df_final_greedy.index = np.arange(1, len(df_final_greedy) + 1)
df_final_numpy_greedy = df_final_greedy.to_numpy()

greedy_list = []
for i in range(len(df_final_numpy_greedy)):
    greedy_list.append(float(df_final_numpy_greedy[i]))

greedy_dictionary = {}
for i in range(len(greedy_list)):
    greedy_dictionary[i+1] = greedy_list[i]
sorted_dictionary = sorted(greedy_dictionary.items(), key=lambda kv: kv[1])
ordered_dictionary = collections.OrderedDict(sorted_dictionary)
print(ordered_dictionary)

ordered_dictionary_new = {key:val for key, val in ordered_dictionary.items() if val == 1}
print(ordered_dictionary_new)


list_keys_od = list(ordered_dictionary_new.keys())
list_values_od = list(ordered_dictionary_new.values())


seed_list_selected = list_keys_od

data_frame_list = [[] for _ in range(len(seed_list_selected))]
for i, item in enumerate(seed_list_selected):
    data_frame_list[i]=pd.read_csv(path + 
                                    str(item).zfill(4) + 
                                    '/greedy_output/'+
                                    'check_position_heading'+
                                    '-NUM_AGENT'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed'+
                                    str(random_seed).zfill(4)+
                                    '-NUMBER_OF_AGENT_TRAINED_MODEL'+
                                    str(Number_agent).zfill(2)+
                                    '-Random_seed_trained_model'+
                                    str(item).zfill(4) +
                                    '.csv',header=None)
label_list = []
for i in range(Number_agent):
    label_list.append('agent' + str(i+1))

df_final = data_frame_list[0]
for i in range(len(data_frame_list)-1):
  df_final = pd.concat([df_final, data_frame_list[i+1]], ignore_index=True)
df_final.index = np.arange(1, len(df_final) + 1)
#print(df_final)
#print(df_final.head())
df_final_numpy = df_final.to_numpy()
print(df_final_numpy)
#print(len(df_final_numpy))


color_list = ['r--', 'b-.', 'g:', 'm-', 'y','c', 'tab:blue', 'tab:orange', 'tab:green',
                      'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                      'tab:cyan']





for i in range(len(df_final_numpy)):
    data = list(df_final_numpy[i])
    data1 = data[0]
    data2 = data[1]
    data3 = data[2]
    data1 = int(data1)
    if Number_of_rotates == 0:
        if data1 == 0:
            pg.draw.circle(env.screen, constant.magenta, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 1:
        if data1 == 1 or data1 == -1:
            pg.draw.circle(env.screen, constant.green, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 2:
        if data1 == 2 or data1 == -2:
            pg.draw.circle(env.screen, constant.olive, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 3:
        if data1 == 3 or data1 == -3:
            pg.draw.circle(env.screen, constant.red, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 4:
        if data1 == 4 or data1 == -4:
            pg.draw.circle(env.screen, constant.cyan, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 5:
        if data1 == 5 or data1 == -5:
            pg.draw.circle(env.screen, constant.purple, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 6:
        if data1 == 6 or data1 == -6:
            pg.draw.circle(env.screen, constant.brown, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 7:
        if data1 == 7 or data1 == -7:
            pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 8:
        if data1 == 8 or data1 == -8:
            pg.draw.circle(env.screen, constant.orange, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 9:
        if data1 == 9 or data1 == -9:
            pg.draw.circle(env.screen, constant.maroon, [int(data2),int(data3)], 2)
        else:
            pass
    elif Number_of_rotates == 10:
        if data1 == 10 or data1 == -10:
            pg.draw.circle(env.screen, constant.yellow, [int(data2),int(data3)], 2)
        else:
            pass
    else:
        print('Error')



'''
    if data1 == 0:
        pg.draw.circle(env.screen, constant.magenta, [int(data2),int(data3)], 0)

    elif data1 == 1 or data1 == -1:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    elif data1 == 2 or data1 == -2:
        pg.draw.circle(env.screen, constant.orange, [int(data2),int(data3)], 0)
    elif data1 == 3 or data1 == -3:
        pg.draw.circle(env.screen, constant.blue, [int(data2),int(data3)], 0)
    elif data1 == 4 or data1 == -4:
        pg.draw.circle(env.screen, constant.brown, [int(data2),int(data3)], 0)
    elif data1 == 5 or data1 == -5:
        pg.draw.circle(env.screen, constant.red, [int(data2),int(data3)], 0)
    elif data1 == 6 or data1 == -6:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    elif data1 == 7 or data1 == -7:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    elif data1 == 8 or data1 == -8:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    elif data1 == 9 or data1 == -9:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    elif data1 == 10 or data1 == -10:
        pg.draw.circle(env.screen, constant.black, [int(data2),int(data3)], 0)
    else:
        print("error")
'''   



    

    #print(data)
    #print(data1)
    #pg.draw.circle(env.screen, constant.red, [int(data2),int(data3)], 0)

#while True:

output_path="./screenshot_rotation_x_y/" + str(NUMBER_OF_AGENT).zfill(2)
if not os.path.exists(output_path):
        os.makedirs(output_path)
env.render()

pg.image.save(env.screen, output_path+ "/screenshot_rotation" + "#" + str(Number_agent).zfill(2) + "-" + str(Number_of_rotates).zfill(2)+".jpg")
    #print('game is over')