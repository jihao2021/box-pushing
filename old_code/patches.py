import pygame as pg
import constant
class patches():
    def __init__(self,patch_x, patch_y, color):
        self.x = patch_x
        self.y = patch_y
        self.position_vector = pg.math.Vector2(patch_x, patch_y)
        self.color = color

    def draw_patch(self,screen):
        pg.draw.rect(screen,self.color,(self.x,self.y, 1, 1), 0)