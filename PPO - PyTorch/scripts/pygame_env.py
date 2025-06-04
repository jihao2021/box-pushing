import pygame
import math
import numpy as np
import sys
import random
import gym
from gym import spaces
from gym.utils import seeding
import time

from .Box import Box
from . import utils
from .game_objects import target, obstacle, box_checker
from .pusher import Pusher

random.seed(utils.random_seed)

N_DISCRETE_ACTIONS = 6


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, random_initialization, n_agents):
        super(CustomEnv, self).__init__()

        '''Define action and observation space, They must be gym.space objects'''
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.random_initialization = random_initialization
        self.n_agents = n_agents

        self.Pusher_NN_name_env = []
        for i in range(self.n_agents):
            self.Pusher_NN_name_env.append("Pusher_NN" + str(i + 1))

        high = 1
        self.observation_space = spaces.Box(-high, high, shape=(9,), dtype=np.float32)

        '''Initialize game screen and clock'''
        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Box-pushing")

        '''Setup game screen, size'''
        self.size = (utils.framewidth, utils.frameheight)
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
        '''w1 * reward_goal + w2 * reward_obstacle + w3 * reward_rotation + 
                              w4 * reward_distance + w5 * reward_efficiency'''
        # self.w1 = 0.6
        self.w1 = 0.6 * 1.0
        # self.w1 = 0.6 * 0.5

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
        target1 = target(utils.target_x, utils.target_y, target_radius=20, color=utils.white)
        target1.draw_target(self.screen)
        self.object_targets.append(target1)

        self.object_obstacles = []
        obstacle1 = obstacle(utils.obstacle_x, utils.obstacle_y, obstacle_radius=15, color=utils.red)
        obstacle1.draw_obstacle(self.screen)
        self.object_obstacles.append(obstacle1)

        self.object_patches = []
        patch_list = utils.setup_background(utils.framewidth, utils.frameheight)
        # patch_list = set_up_background_new(utils.framewidth, utils.frameheight)
        self.object_patches = patch_list

        for patch in patch_list:
            patch.draw_patch(self.screen)

        """reset the environment"""
        _ = self.reset()

    def reset(self):
        '''reset boxes and pushers to void'''
        self.object_boxes = []
        self.object_pushers = []

        self.screen.fill(utils.black)

        self.Pusher_name = []
        for i in range(self.n_agents):
            self.Pusher_name.append("Pusher" + str(i + 1))

        '''
        This is independent variable, whether box will deployed randomly or with fixed position
        '''

        if self.random_initialization:
            box1 = Box(initial_x=random.randint(60, 181), initial_y=random.randint(120, 361), width=utils.box_width,
                       height=utils.box_height)
        else:
            box1 = Box(initial_x=120, initial_y=240, width=utils.box_width, height=utils.box_height)

        # box1.draw_polygon(self.screen)
        box1.draw_closed_lines(self.screen)
        self.object_boxes.append(box1)

        '''Add Pusher and append to object_pusher list'''

        for i in range(self.n_agents):
            self.object_pushers.append(Pusher(size=10, name=self.Pusher_name[i]))

        # for i in range(self.n_agents):
        # self.object_pushers[i].draw_polygon(self.screen)

        '''Add box sensor box_checker'''
        self.add_box_checker()
        state = box1.check_state(self, self.Pusher_NN_name_env)

        '''Initialize goal, collision and done state'''
        self.reach_goal = False
        self.collision = False
        self.done = False

        return state

    def add_box_checker(self):
        '''initialize box checkers'''
        self.object_box_checkers = []

        '''left part of the box checker'''
        x_offset = -utils.box_width / 2
        y_offset = utils.box_height / 2
        for _ in range(9):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            y_offset -= utils.box_height / 8
        x_offset = utils.box_width / 2
        y_offset = utils.box_height / 2

        '''right part of the box checker'''
        for _ in range(9):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            y_offset -= utils.box_height / 8
        y_offset = utils.box_height / 2
        x_offset = - utils.box_width / 4

        '''top part of the box checker'''
        for _ in range(3):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            x_offset += utils.box_width / 4
        y_offset = - utils.box_height / 2
        x_offset = - utils.box_width / 4

        '''bottom part of the box checker'''
        for _ in range(3):
            self.object_box_checkers.append(box_checker(int(x_offset), int(y_offset), self.object_boxes[0]))
            x_offset += utils.box_width / 4

    def render(self):
        '''Get box and pushers ready'''
        box1 = self.object_boxes[0]

        '''Get target and obstacle, patches ready'''
        target1 = self.object_targets[0]
        obstacle1 = self.object_obstacles[0]
        patch_list = self.object_patches

        # canvas = pygame.Surface(self.size)
        # canvas.fill((0, 0, 0))

        # This is the line of code that keeps game from not updating
        # self.screen.fill(utils.black)

        # box1.draw_polygon(self.screen)
        box1.draw_closed_lines(self.screen)

        # for i in range(self.n_agents):
        # self.object_pushers[i].draw_polygon(self.screen)

        # box1.draw_circle(self.screen)
        # box1.draw_line(self.screen)
        target1.draw_target(self.screen)
        obstacle1.draw_obstacle(self.screen)

        pygame.draw.line(self.screen, utils.black, [440, 240],
                         [460, 240], 5)
        pygame.draw.line(self.screen, utils.black, [450, 230],
                         [450, 250], 5)

        for patch in patch_list:
            patch.draw_patch(self.screen)

        # for box_checker in self.object_box_checkers:
        # box_checker.draw_box_checker(self.screen)

        # time.sleep(0.1)
        # self.screen.blit(self.screen, self.screen.get_rect())
        # pygame.event.pump()
        pygame.display.update()
        self.clock.tick(utils.FPS)

        # test canvas
        # import cv2
        # cv2.imwrite("test.png", np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     ))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()
                # sys.exit()

    def check_reward_state(self, Pusher_NN_name):
        box1 = self.object_boxes[0]

        '''Get reward_goal, reward_obstalce and done information'''
        reward_goal, reward_obstacle, done = self.check_goal_and_collision_reward_new()

        '''Check rotation reward and distance reward'''
        reward_rotation = box1.check_rotation_reward()
        reward_distance = box1.check_distance_reward(self)

        '''Add reward together first part of reward'''
        reward = []
        for i in range(self.n_agents):
            reward.append(self.w1 * reward_goal + self.w2 * reward_obstacle + self.w3 * reward_rotation + \
                          self.w4 * reward_distance)

        reward_efficiency = utils.reward_efficiency

        for i in range(self.n_agents):
            reward[i] += self.w5 * reward_efficiency[i]

        reward_dict = {}
        for i in range(self.n_agents):
            reward_dict.update({Pusher_NN_name[i]: reward[i]})

        done_dict = {}
        for i in range(self.n_agents):
            done_dict.update({Pusher_NN_name[i]: done})
        return reward_dict, done_dict

    def check_goal_and_collision_reward_new(self):
        """
        Check reward_goal and whether is it done or not
        """
        target1 = self.object_targets[0]
        for box_checker in self.object_box_checkers:
            # if box_checker.x > (target1.x - 5):
            if box_checker.x > (target1.x + 15):
                self.reach_goal = True

                print("Reached Goal, yay!")
                break

        if self.reach_goal:
            reward_goal = 900
            print("reach_goal_reward_get")
        else:
            # reward_goal = -1
            '''Starting seven actions 2/14/2019, reward_goal will not give negative -1 to it'''
            reward_goal = 0

        for obstacle in self.object_obstacles:
            for box_checker in self.object_box_checkers:
                if math.hypot(box_checker.x - obstacle.x, box_checker.y - obstacle.y) < 15:
                    self.collision = True
                    print("Collision with obstacle")
                    break
            if self.collision:
                break

        for patch in self.object_patches:
            for box_checker in self.object_box_checkers:
                if math.hypot(box_checker.x - patch.x, box_checker.y - patch.y) < 5:
                    self.collision = True
                    print("Collision with patches")
                    break
            if self.collision:
                break
        if self.collision:
            reward_obstacle = -900
            # pygame.image.save(self.screen, "screenshot.jpg")
        else:
            reward_obstacle = 0

        if self.reach_goal or self.collision:
            self.done = True
        return reward_goal, reward_obstacle, self.done

    def step(self, action, Pusher_NN_name):
        box1 = self.object_boxes[0]

        # Can simulate up to 10 agents?
        for i in range(self.n_agents):
            for key, value in action.items():
                '''Check if the last letter is the same or not'''
                if key[-1] == self.object_pushers[i].name[-1]:
                    action_pusher = action[key]
            self.object_pushers[i].move(action_pusher, box1)

        box1.move(action, Pusher_NN_name)

        self.render()

        '''Newly added box_checker movement'''
        for box_checker in self.object_box_checkers:
            box_checker.determine_coordinate(box1)

        reward, done = self.check_reward_state(Pusher_NN_name)
        state = box1.check_state(self, Pusher_NN_name)

        '''information splitted up to three different training agents'''

        information = "Welcome to multi-agent world, full of challenge and fun, fight on"

        info = {}
        for i in range(self.n_agents):
            info.update({'Pusher_NN' + str(i + 1): information})
        return state, reward, done, info
