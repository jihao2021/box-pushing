#Box_checker is just to draw a circle of radius to show detection range
import math
import random
import pygame as pg
import numpy as np
from time import sleep
from . import utils

class obstacle():
    def __init__(self,obstacle_x,obstacle_y,obstacle_radius,color):
        self.x = obstacle_x
        self.y = obstacle_y
        self.radius = obstacle_radius
        self.color = color
        self.position_vector = pg.math.Vector2(obstacle_x, obstacle_y)

    def draw_obstacle(self,screen):
        pg.draw.circle(screen,self.color,(self.x,self.y),self.radius,0)


class target():
    def __init__(self,target_x,target_y,target_radius, color):
        self.radius = target_radius
        self.x = target_x
        self.y = target_y
        self.color = color
        self.position_vector = pg.math.Vector2(target_x, target_y)

    def draw_target(self,screen):
        pg.draw.circle(screen,self.color,(self.x,self.y),self.radius, 0)


class patches():
    def __init__(self,patch_x, patch_y, color):
        self.x = patch_x
        self.y = patch_y
        self.position_vector = pg.math.Vector2(patch_x, patch_y)
        self.color = color

    def draw_patch(self,screen):
        pg.draw.rect(screen,self.color,(self.x,self.y, 1, 1), 0)


class box_checker():
    def __init__(self,  x_offset, y_offset, Box):
        self.x = Box.position_vector.x + x_offset
        self.y = Box.position_vector.y + y_offset
        self.current_heading = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.color =(0, 0, 255)   ##green color
        self.radius = 1

    def determine_coordinate(self, Box):
        Box1 = Box
        self.x,self.y = utils.local_to_global_transform(Box1.position_vector.x, Box1.position_vector.y, self.x_offset, self.y_offset, Box1.current_heading)
        self.current_heading = Box1.current_heading

    def draw_box_checker(self,screen):
        pg.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius, 0)






