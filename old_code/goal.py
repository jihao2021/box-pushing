import pygame as pg

import constant


class goal():
    def __init__(self,goal_x,goal_y, color):
        self.x = goal_x
        self.y = goal_y
        self.color = color
        self.position_vector = pg.math.Vector2(goal_x, goal_y)

    def draw_goal(self,screen):
        #pg.draw.circle(screen,self.color,(self.x,self.y),self.radius,0)
        pg.draw.line(screen, self.color, (goal_x, int(constant.frame_height/4), (goal_x, int(constant.frame_height/4*3), 2)