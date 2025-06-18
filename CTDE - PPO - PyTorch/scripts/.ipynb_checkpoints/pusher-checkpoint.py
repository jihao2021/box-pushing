import pygame as pg
import math
import numpy as np
import random

from . import utils
from .Pusher_NN_PPO_CTDE import Pusher_NN_PPO_CTDE

random.seed(6)

class Pusher():
    def __init__(self, size, name):
        self.initial_x = random.randint(30, 90)
        self.initial_y = random.randint(160, 320)
        if name == "Pusher1":
            self.initial_x = 40
            self.initial_y = 180
       
        if name == "Pusher2":
            self.initial_x = 60
            self.initial_y = 250

        self.name = name
        self.size = size
        self.initial_position_vector = pg.math.Vector2(self.initial_x, self.initial_y)
        self.position_vector = pg.math.Vector2(self.initial_x, self.initial_y)
        self.velocity_vector = pg.math.Vector2(0, 0)
        self.heading_vector = pg.math.Vector2(1, 0)
        _, self.current_heading = self.heading_vector.as_polar()

        self.past_action = None
        self.distance_travelled = 0

        self.left_top_x = None
        self.left_top_y = None
        self.right_top_x = None
        self.right_top_y = None
        self.right_down_x = None
        self.right_down_y = None
        self.left_down_x = None
        self.left_down_y = None

        self.left_top_change_x = -self.size / 2
        self.left_top_change_y = self.size / 2

        self.right_top_change_x = self.size / 2
        self.right_top_change_y = self.size / 2

        self.right_down_change_x = self.size / 2
        self.right_down_change_y = -self.size / 2

        self.left_down_change_x = -self.size / 2
        self.left_down_change_y = -self.size / 2

        self.distance_travelled_history_each_episode = []
        
    def move(self, action, Box1):
        self.heading_vector = Box1.heading_vector

        '''initialize x_random and y_random and this is the relative coordinate of the pusher to box center'''

        if action == 0:
            x_random = utils.box_width / 2 + self.size / 2
            y_random = utils.box_height / 4 + random.randint(-10, 10)
            self.velocity_vector = x_random * self.heading_vector + \
                                   y_random * self.heading_vector.rotate(90)

            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 1:
            x_random = utils.box_width / 2 + self.size / 2
            y_random = utils.box_height / 4 + random.randint(-10, 10)
            self.velocity_vector= x_random * self.heading_vector - y_random * self.heading_vector.rotate(90)
            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 2:
            x_random = utils.box_width / 2 + self.size / 2
            y_random = utils.box_height / 4 + random.randint(-10, 10)
            self.velocity_vector = -x_random * self.heading_vector - y_random * self.heading_vector.rotate(90)
            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 3:
            x_random = utils.box_width / 2 + self.size / 2
            y_random = utils.box_height / 4 + random.randint(-10, 10)
            self.velocity_vector = x_random * self.heading_vector + y_random * self.heading_vector.rotate(90)
            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 4:
            x_random = 0 + random.randint(-10,10)
            y_random = utils.box_height / 2 + self.size / 2
            self.velocity_vector = x_random * self.heading_vector + y_random * self.heading_vector.rotate(90)
            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 5:
            x_random = 0 + random.randint(-10, 10)
            y_random = utils.box_height / 2 + self.size / 2
            self.velocity_vector = x_random * self.heading_vector - y_random * self.heading_vector.rotate(90)
            self.position_vector = Box1.position_vector + self.velocity_vector

        if action == 6:
            pass

        _, self.current_heading = Box1.heading_vector.as_polar()
        self.calculate_distance(action)


    def draw_polygon(self, screen):
        self.determine_vertices()
        pg.draw.polygon(screen, utils.green, ((self.left_top_x, self.left_top_y),
                                                 (self.right_top_x, self.right_top_y),
                                                 (self.right_down_x, self.right_down_y),
                                                 (self.left_down_x, self.left_down_y)))

    def determine_vertices(self):
        _, self.current_heading = self.heading_vector.as_polar()


        self.left_top_x, self.left_top_y = utils.local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                     self.left_top_change_x, self.left_top_change_y,
                                                                     self.current_heading)
        self.right_top_x, self.right_top_y = utils.local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                       self.right_top_change_x, self.right_top_change_y,
                                                                       self.current_heading)
        self.right_down_x, self.right_down_y = utils.local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                         self.right_down_change_x,
                                                                         self.right_down_change_y, self.current_heading)
        self.left_down_x, self.left_down_y = utils.local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                       self.left_down_change_x, self.left_down_change_y,
                                                                       self.current_heading)

    def calculate_distance(self, action):
        distance = None
        if self.past_action == None:
            distance = 0
            if action == 6:
                pass
            else:
                self.past_action = action
        else:
            if self.past_action == 0:
                if action == 0:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0

                else:
                    print("Error occurred in calculate_distance function in pusher module")

            elif self.past_action == 1:
                if action == 0:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0

                else:
                    print("Error occurred in calculate_distance function in pusher module")

            elif self.past_action == 2:
                if action == 0:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0

                else:
                    print("Error occurred in calculate_distance function in pusher module")

            elif self.past_action == 3:
                if action == 0:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0

                else:
                    print("Error occurred in calculate_distance function in pusher module")

            elif self.past_action == 4:
                if action == 0:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0

                else:
                    print("Error occurred in calculate_distance function in pusher module")

            elif self.past_action == 5:
                if action == 0:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 1:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 2:
                    self.distance_travelled += 1
                    distance = 1
                    self.past_action = action

                elif action == 3:
                    self.distance_travelled += 2
                    distance = 2
                    self.past_action = action

                elif action == 4:
                    self.distance_travelled += 3
                    distance = 3
                    self.past_action = action

                elif action == 5:
                    self.distance_travelled += 0
                    distance = 0
                    self.past_action = action

                elif action == 6:
                    self.distance_travelled += 0
                    distance = 0
                else:
                    print("Error occurred in calculate_distance function in pusher module")
            else:
                print("Error occurred in calculate_distance function in pusher module")
            self.distance_travelled_history_each_episode.append(distance)
        return distance