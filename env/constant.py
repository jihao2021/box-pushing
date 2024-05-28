import pygame
"""Setup game screen and size"""
framewidth = 600
frameheight = 480

'''Setup target location'''
target_x = 450
target_y= 240

'''Setup obstacle location'''
obstacle_x = 270
obstacle_y = 240

'''Other parameters'''



FPS = 480
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255,0)
black = (0,0,0)
brown = (138,54,15)
magenta = (255,0,255)
olive=(128,128,0)
orange=(255,69,0)
cyan = (0,255,255)
purple = (128,0,128)
maroon=(128,0,0)
random_seed = 4

'''Setup box width and height'''
box_width = 60
box_height = 150

#box_width = 90
#box_height = 225

distance_coefficient = 2.5

reward_efficiency = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

color_list = ['r--', 'b-.', 'g:', 'm-', 'y','c', 'tab:blue', 'tab:orange', 'tab:green',
                      'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                      'tab:cyan']







