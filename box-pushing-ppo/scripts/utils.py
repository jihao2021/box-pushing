from .game_objects import patches
import math
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

def local_to_global_transform(current_x, current_y, local_x, local_y, heading):
    updated_current_x = local_x * math.cos(math.radians(heading)) - local_y * math.sin(math.radians(heading)) + current_x
    updated_current_y = local_x * math.sin(math.radians(heading)) + local_y * math.cos(math.radians(heading)) + current_y
    return updated_current_x, updated_current_y

def global_to_local_transform(current_x, current_y, global_x, global_y, heading):
    updated_local_x = (global_x - current_x) * math.cos(math.radians(heading)) + (global_y - current_y) * math.sin(math.radians(heading))
    updated_local_y = - (global_x - current_x) * math.sin(math.radians(heading)) + (global_y - current_y) * math.cos(math.radians(heading))
    return updated_local_x, updated_local_y

def setup_background(background_width,background_height, color = white):
    patch_list1 = []
    patch_list2 = []
    patch_list3 = []
    patch_list4 = []
    patch_list5 = []
    patch_list6 = []
    patch_list7 = []
    patch_list = []
    patch_color = color
    for x in range(0,background_width+1):
        for y in range(0,background_height+1):
            if x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20:
                patch_list1.append(patches(x,y,color=patch_color))
            elif x >= background_width / 30 and x <= background_width / 2 and y == background_height / 20 * 19:
                patch_list2.append(patches(x, y, color=patch_color))
            elif x == background_width / 30 and y >= background_height / 20 and y <= background_height / 20 * 19:
                patch_list3.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(16 / 25 * x - 168):
                patch_list4.append(patches(x, y, color=patch_color))
            elif x >= background_width / 2 and x <= 3 / 4 * background_width and y == int(- 16 / 25 * x + 648):
                patch_list5.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 1 / 4:
                patch_list6.append(patches(x, y, color=patch_color))
            elif x >= 3 / 4 * background_width and x <= background_width and y == background_height * 3 / 4:
                patch_list7.append(patches(x, y, color=patch_color))
            else:
                pass

    patch_list += patch_list1
    patch_list += patch_list2
    patch_list += patch_list3
    patch_list += patch_list4
    patch_list += patch_list5
    patch_list += patch_list6
    patch_list += patch_list7

    patch_list_new = patch_list[::10]

    return patch_list_new