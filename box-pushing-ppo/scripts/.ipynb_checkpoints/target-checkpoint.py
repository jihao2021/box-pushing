from . import constant
import pygame as pg

class target():
    def __init__(self,target_x,target_y,target_radius, color):
        self.radius = target_radius
        self.x = target_x
        self.y = target_y
        self.color = color
        self.position_vector = pg.math.Vector2(target_x, target_y)

    def draw_target(self,screen):
        pg.draw.circle(screen,self.color,(self.x,self.y),self.radius, 0)