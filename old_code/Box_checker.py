#Box_checker is just to draw a circle of radius to show detection range
import math
import random
import pygame
import numpy as np
from time import sleep
import constant
from coordinate_tranformation import local_to_global_transform,global_to_local_transform

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
        self.x,self.y = local_to_global_transform(Box1.position_vector.x, Box1.position_vector.y, self.x_offset, self.y_offset, Box1.current_heading)
        self.current_heading = Box1.current_heading

    def draw_box_checker(self,screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius, 0)



