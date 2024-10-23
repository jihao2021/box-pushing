import math
import pygame as pg
import numpy as np
from . import constant
from .coordinate_tranformation import local_to_global_transform, global_to_local_transform

#vec = pygame.math.Vector2
'''For future version update, simplify vector '''

"""Define Class Box:"""
class Box():
    def __init__(self, initial_x, initial_y,  width, height):
        self.speed_factor = 1/5 #CHANGE AS NEEDED

        self.width = width
        self.height = height
        self.initial_position_vector = pg.math.Vector2(initial_x,initial_y)
        self.position_vector = pg.math.Vector2(initial_x, initial_y)
        self.velocity_vector = pg.math.Vector2(0,0)
        self.heading_vector = pg.math.Vector2(1,0)
        _, self.current_heading = self.heading_vector.as_polar()

        self.one_hit = 0   ##top hit
        self.two_hit = 0   ##right-top corner hit
        self.three_hit = 0  ##right-down corner hit
        self.four_hit = 0  ##down hit
        self.five_hit = 0  ##left-down corner hit
        self.six_hit = 0  ##left-top corner hit

        self.change_x_self = 0  ## change of x with respect to local coordinate
        self.change_y_self = 0  ##change of y with respect to local coordinate

        self.change_x = 0   ## change of x with respect to global coordinate
        self.change_y = 0   ## change of x with respect to global coordinate
        self.change_angle = 0

        #the coordinate of four vertices
        self.left_top_x = None
        self.left_top_y = None
        self.right_top_x = None
        self.right_top_y = None
        self.right_down_x = None
        self.right_down_y = None
        self.left_down_x = None
        self.left_down_y = None

        self.left_top_change_x = -self.width / 2
        self.left_top_change_y = self.height / 2

        self.right_top_change_x = self.width / 2
        self.right_top_change_y = self.height / 2

        self.right_down_change_x = self.width / 2
        self.right_down_change_y = -self.height / 2

        self.left_down_change_x = -self.width / 2
        self.left_down_change_y = -self.height / 2

        self.previous_distance = -100
        self.current_distance = None

    def move(self, actions, Boxer_name):
        self.determine_total_hit(actions, Boxer_name)
        self.determine_move()
        self.determine_vertices()

    def determine_total_hit(self, action, Boxer_name):
        #print(action)
        self.one_hit = 0
        self.two_hit = 0
        self.three_hit = 0
        self.four_hit = 0
        self.five_hit = 0
        self.six_hit = 0
        actions = []
        for i in range(len(action)):
            actions.append(action[Boxer_name[i]])

        for i in range(len(actions)):
            if actions[i] == 0:
                self.one_hit += 1
            elif actions[i] == 1:
                self.two_hit += 1
            elif actions[i] ==2:
                self.three_hit += 1
            elif actions[i] == 3:
                self.four_hit += 1
            elif actions[i] == 4:
                self.five_hit += 1
            elif actions[i] == 5:
                self.six_hit += 1
            elif actions[i] == 6:
                pass
            else:
                print("Error in Box determine_total_hit")
            
    """Calculate box movement with respect to box local coordinate system"""
    def determine_move(self):
        #print(self.one_hit, self.two_hit, self.three_hit, self.four_hit, self.five_hit, self.six_hit)
        self.change_x_self = (self.five_hit + self.six_hit - self.two_hit - self.three_hit) / 1 * 5 * self.speed_factor
        self.change_y_self = (self.four_hit - self.one_hit) / 1 * 5 * self.speed_factor
        self.velocity_vector = self.change_x_self * self.heading_vector + self.change_y_self * self.heading_vector.rotate(90)
        self.position_vector += self.velocity_vector
        self.change_angle = (self.two_hit  + self.five_hit - self.three_hit - self.six_hit) / 1 * 5 * self.speed_factor
        self.heading_vector = self.heading_vector.rotate(self.change_angle)
        _, self.current_heading = self.heading_vector.as_polar()

        

    def determine_vertices(self):
        _, self.current_heading = self.heading_vector.as_polar()


        self.left_top_x, self.left_top_y = local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                     self.left_top_change_x, self.left_top_change_y,
                                                                     self.current_heading)
        self.right_top_x, self.right_top_y = local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                       self.right_top_change_x, self.right_top_change_y,
                                                                       self.current_heading)
        self.right_down_x, self.right_down_y = local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                         self.right_down_change_x,
                                                                         self.right_down_change_y, self.current_heading)
        self.left_down_x, self.left_down_y = local_to_global_transform(self.position_vector.x, self.position_vector.y,
                                                                       self.left_down_change_x, self.left_down_change_y,
                                                                       self.current_heading)

    def draw_polygon(self,screen):
        self.determine_vertices()
        pg.draw.polygon(screen, constant.brown,((self.left_top_x,self.left_top_y),(self.right_top_x,self.right_top_y),
                                                (self.right_down_x, self.right_down_y),(self.left_down_x, self.left_down_y)))

    def draw_line(self, screen):
        pg.draw.line(screen, constant.red, [self.position_vector.x, self.position_vector.y],
                     [(self.position_vector.x + self.heading_vector.x * 30),
                      (self.position_vector.y + self.heading_vector.y * 30)], 5)



    def draw_closed_lines(self,screen):
        self.determine_vertices()
        pg.draw.polygon(screen, constant.magenta,
                        ((self.left_top_x, self.left_top_y), (self.right_top_x, self.right_top_y),
                         (self.right_down_x, self.right_down_y), (self.left_down_x, self.left_down_y)),1)


    def check_state(self, env, Boxer_name):
        
        state = np.zeros(env.observation_space.shape[0])

        obstacle1 = env.object_obstacles[0]
        target1 = env.object_targets[0]
        patches = env.object_patches

        obstacle_x, obstacle_y = obstacle1.position_vector.x, obstacle1.position_vector.y
        target_x, target_y = target1.position_vector.x, target1.position_vector.y
        patch_list = patches
        _, self.current_heading = self.heading_vector.as_polar()



        if math.hypot(self.position_vector.x - obstacle_x, self.position_vector.y - obstacle_y) < 150:
            relative_x_obstacle, relative_y_obstacle = global_to_local_transform(self.position_vector.x, self.position_vector.y,
                                                                                 obstacle_x, obstacle_y,
                                                                                 self.current_heading)
            angle_obstacle = math.degrees(math.atan2(relative_y_obstacle, relative_x_obstacle))

            if 0 <= angle_obstacle <= 180:
                pass
            else:
                angle_obstacle += 360
            if 0 <= angle_obstacle < 360:
                pass
            elif angle_obstacle == 360:
                angle_obstacle = 0
            else:
                print("angle_obstacle out of angle bound angle_obstacle")


            assert (0 <= angle_obstacle < 360), "angle_obstacle is incorrect"

            if 0 <= angle_obstacle < 45:
                state[0] = 1
            elif 45 <= angle_obstacle < 90:
                state[1] = 1
            elif 90 <= angle_obstacle < 135:
                state[2] = 1
            elif 135 <= angle_obstacle < 180:
                state[3] = 1
            elif 180 <= angle_obstacle < 225:
                state[4] = 1
            elif 225 <= angle_obstacle < 270:
                state[5] = 1
            elif 270 <= angle_obstacle < 315:
                state[6] = 1
            elif 315 <= angle_obstacle < 360:
                state[7] = 1
            else:
                print("Error")
        else:
            pass

        for patch in patch_list:
            patch_x = patch.position_vector.x
            patch_y = patch.position_vector.y
            if math.hypot(self.position_vector.x - patch_x, self.position_vector.y - patch_y) < 150:

                relative_x_patch, relative_y_patch = global_to_local_transform(self.position_vector.x,
                                                                               self.position_vector.y, patch_x,
                                                                              patch_y, self.current_heading)
                angle_patch = math.degrees(math.atan2(relative_y_patch, relative_x_patch))
                if 0 <= angle_patch <= 180:
                    pass
                else:
                    angle_patch += 360
                if 0 <= angle_patch < 360:
                    pass
                elif angle_patch == 360:
                    angle_patch = 0
                else:
                    print("angle_patch out of angle bound angle_obstacle")



                assert(0 <= angle_patch < 360), "angle_patch is incorrect"

                if 0 <= angle_patch < 45:
                    state[0] = 1
                elif 45 <= angle_patch < 90:
                    state[1] = 1
                elif 90 <= angle_patch < 135:
                    state[2] = 1
                elif 135 <= angle_patch < 180:
                    state[3] = 1
                elif 180 <= angle_patch < 225:
                    state[4] = 1
                elif 225 <= angle_patch < 270:
                    state[5] = 1
                elif 270 <= angle_patch < 315:
                    state[6] = 1
                elif 315 <= angle_patch < 360:
                    state[7] = 1
                else:
                    print("Error in Check_state angle_patch")
            else:
                pass
        relative_x_goal, relative_y_goal = global_to_local_transform(self.position_vector.x, self.position_vector.y,
                                                                     target_x, target_y, self.current_heading)

        angle_goal = math.degrees(math.atan2(relative_y_goal, relative_x_goal))

        if 0 <= angle_goal <= 360:
            pass
        else:
            angle_goal += 360
        if 0 <= angle_goal < 360:
            pass
        elif angle_goal == 360:
            angle_goal = 0
        else:
            print("angle_goal out of angle bound angle_obstacle")
        assert (0 <= angle_goal < 360), "angle_goal is incorrect"

        '''
        Will add this Jan 22nd 2019 and comment the code under this 
        '''
        angle_goal = (angle_goal - 180) / 180
        state[8] = angle_goal

        state = np.array(state)
        
        state_dict = {}
        for i in range(len(Boxer_name)):
            state_dict.update({Boxer_name[i]: state})

        return state_dict

    def draw_circle(self,screen):
        pg.draw.circle(screen,constant.blue, (int(self.position_vector.x), int(self.position_vector.y)), 150,  int(1))


    def check_distance_reward(self, env):
        target1 = env.object_targets[0]
        if self.previous_distance == -100:
            self.previous_distance = math.hypot(self.initial_position_vector.x - target1.x,
                                            self.initial_position_vector.y - target1.y)
        else:
            pass

        self.current_distance = math.hypot(self.position_vector.x - target1.x, self.position_vector.y - target1.y)
        change_distance = self.previous_distance - self.current_distance
        reward_distance = change_distance * constant.distance_coefficient
        self.previous_distance = self.current_distance
        return reward_distance

    def check_rotation_reward(self):
        return 0
        change_angle = self.change_angle
        reward_rotation = math.cos(math.radians(change_angle)) - 0.98
        return reward_rotation
        














