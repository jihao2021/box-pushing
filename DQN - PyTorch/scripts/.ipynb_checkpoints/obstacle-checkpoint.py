import pygame as pg
from . import constant


class obstacle():
    def __init__(self,obstacle_x,obstacle_y,obstacle_radius,color):
        self.x = obstacle_x
        self.y = obstacle_y
        self.radius = obstacle_radius
        self.color = color
        self.position_vector = pg.math.Vector2(obstacle_x, obstacle_y)

    def draw_obstacle(self,screen):
        pg.draw.circle(screen,self.color,(self.x,self.y),self.radius,0)


