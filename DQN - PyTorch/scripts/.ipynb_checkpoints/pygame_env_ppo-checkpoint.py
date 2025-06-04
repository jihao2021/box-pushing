import pygame
import math
import numpy as np
import random
import gym
from gym import spaces

from .Box import Box
from . import utils
from .game_objects import target, obstacle, box_checker

random.seed(utils.random_seed)

N_DISCRETE_ACTIONS = 6


class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, random_initialization, n_agents):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.random_initialization = random_initialization
        self.n_agents = n_agents
        # agent identifiers
        self.agent_names = [f"Pusher_NN{i+1}" for i in range(self.n_agents)]

        high = 1.0
        self.observation_space = spaces.Box(-high, high, shape=(9,), dtype=np.float32)

        pygame.init()
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Box-pushing PPO")
        self.size = (utils.framewidth, utils.frameheight)
        self.screen = pygame.display.set_mode(self.size)

        # reward weights
        self.reward_rotation = True
        self.w1 = 0.6 * 1.0
        self.w2 = 0.1
        self.w3 = 0.1 if self.reward_rotation else 0.0
        self.w4 = 0.2
        self.w5 = 0
        self.w6 = 0

        # static objects
        self.object_targets = []
        tgt = target(utils.target_x, utils.target_y, target_radius=20, color=utils.white)
        self.object_targets.append(tgt)

        self.object_obstacles = []
        obs = obstacle(utils.obstacle_x, utils.obstacle_y, obstacle_radius=15, color=utils.red)
        self.object_obstacles.append(obs)

        self.object_patches = utils.setup_background(utils.framewidth, utils.frameheight)

        # dynamic
        self.object_boxes = []
        self.object_pushers = []
        self.object_box_checkers = []

        _ = self.reset()

    def reset(self):
        self.object_boxes.clear()
        self.object_pushers.clear()
        self.screen.fill(utils.black)

        # create box
        if self.random_initialization:
            x = random.randint(60, 181)
            y = random.randint(120, 361)
        else:
            x, y = 120, 240
        box1 = Box(initial_x=x, initial_y=y, width=utils.box_width, height=utils.box_height)
        box1.draw_closed_lines(self.screen)
        self.object_boxes.append(box1)

        # create pushers
        for i in range(self.n_agents):
            self.object_pushers.append(Pusher(size=10, name=f"Pusher{i+1}"))

        # setup box checkers
        self.add_box_checker()

        self.reach_goal = False
        self.collision = False
        self.done = False

        # initial observation
        obs = box1.check_state(self, self.agent_names)
        return obs

    def add_box_checker(self):
        self.object_box_checkers.clear()
        # left/right
        offsets = [(-utils.box_width/2, utils.box_height/2), (utils.box_width/2, utils.box_height/2)]
        for x_off, y_start in offsets:
            y = y_start
            for _ in range(9):
                self.object_box_checkers.append(box_checker(int(x_off), int(y), self.object_boxes[0]))
                y -= utils.box_height/8
        # top/bottom
        y_offs = (utils.box_height/2, -utils.box_height/2)
        for y_off in y_offs:
            x = -utils.box_width/4
            for _ in range(3):
                self.object_box_checkers.append(box_checker(int(x), int(y_off), self.object_boxes[0]))
                x += utils.box_width/4

    def render(self):
        box1 = self.object_boxes[0]
        tgt = self.object_targets[0]
        obs = self.object_obstacles[0]
        # clear
        self.screen.fill(utils.black)
        # draw
        box1.draw_closed_lines(self.screen)
        tgt.draw_target(self.screen)
        obs.draw_obstacle(self.screen)
        # center crosshair
        pygame.draw.line(self.screen, utils.black, [440,240],[460,240],5)
        pygame.draw.line(self.screen, utils.black, [450,230],[450,250],5)
        for patch in self.object_patches:
            patch.draw_patch(self.screen)
        pygame.display.update()
        self.clock.tick(utils.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
                pygame.quit()

    def step(self, actions: dict):
        box1 = self.object_boxes[0]
        # apply each agent's action
        for i, pusher in enumerate(self.object_pushers):
            name = self.agent_names[i]
            act = actions.get(name, 0)
            pusher.move(act, box1)
        # box movement if needed
        box1.move(actions, self.agent_names)

        self.render()
        for checker in self.object_box_checkers:
            checker.determine_coordinate(self.object_boxes[0])

        rewards, dones = self.check_reward_state()
        obs = self.object_boxes[0].check_state(self, self.agent_names)

        # Gym API expects scalar done for single-agent; here we return dict
        return obs, rewards, dones, {}

    def check_reward_state(self):
        box1 = self.object_boxes[0]
        reward_goal, reward_obstacle, done_flag = self.check_goal_and_collision_reward_new()
        reward_rotation = box1.check_rotation_reward()
        reward_distance = box1.check_distance_reward(self)

        # assemble
        rewards = {}
        dones = {}
        for i, name in enumerate(self.agent_names):
            r = self.w1*reward_goal + self.w2*reward_obstacle + self.w3*reward_rotation + self.w4*reward_distance
            r += self.w5 * utils.reward_efficiency[i]
            rewards[name] = r
            dones[name] = done_flag
        return rewards, dones

    def check_goal_and_collision_reward_new(self):
        target1 = self.object_targets[0]
        for checker in self.object_box_checkers:
            if checker.x > (target1.x + 15):
                self.reach_goal = True
                break
        reward_goal = 900 if self.reach_goal else 0

        for obstacle in self.object_obstacles:
            for checker in self.object_box_checkers:
                if math.hypot(checker.x - obstacle.x, checker.y - obstacle.y) < 15:
                    self.collision = True
                    break
            if self.collision:
                break
        reward_obstacle = -900 if self.collision else 0

        self.done = self.reach_goal or self.collision
        return reward_goal, reward_obstacle, self.done
