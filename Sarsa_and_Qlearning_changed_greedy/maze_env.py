"""
Reinforcement learning Robot Frozen Lake.
Red rectangle:          explorer robot
Blue rectangles:        holes       [reward = -1]
Yellow bin circle:      goals       [reward = +1]
All other states:       ground      [reward = 0]
If MonteCarlo environment: ground   [reward = -0.04]
The creation of RL environment is referred to Zhou-Mofan's code: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import time
import sys
import random
import tkinter as tk

UNIT = 80   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
size_of_element = UNIT//2-UNIT//20
holes_location = []
goal_location = []
goal_set = [7, 2]

class Maze(tk.Tk, object):

    def __init__(self, height, width):
        super(Maze, self).__init__()
        global MAZE_W
        global MAZE_H
        self.height = height
        self.width = width
        MAZE_H = self.height
        MAZE_W = self.width
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()
    # Use Tkinter to create the canvas environment(the back white board)

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        # create grids of the canvas
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        # hole
        locations = [
            [5, 0], [6, 0], [9, 0],
            [0, 1], [2, 1], [6, 1], [9, 1],
            [0, 2], [2, 2],
            [4, 3], [6, 3],
            [1, 4],
            [1, 5], [3, 5], [5, 5], [6, 5],
            [3, 6], [8, 6], [9, 6],
            [1, 7],
            [1, 8], [3, 8], [4, 8], [8, 8],
            [4, 9]
        ]
        self.creat_holes_directly(locations)
        # self.create_holes_randomly(25)
        # create goal
        oval_center = []
        origin = np.array([UNIT * 0.5, UNIT * 0.5])
        oval_center.append(origin[0] + UNIT * goal_set[0])
        oval_center.append(origin[1] + UNIT * goal_set[1])
        self.oval = self.canvas.create_oval(
            oval_center[0] - size_of_element, oval_center[1] - size_of_element,
            oval_center[0] + size_of_element, oval_center[1] + size_of_element,
            fill='yellow')
        goal_location.append(self.canvas.coords(self.oval))

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - size_of_element, origin[1] - size_of_element,
            origin[0] + size_of_element, origin[1] + size_of_element,
            fill='red')
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        self.canvas.delete(self.rect)
        origin = np.array([UNIT * 0.5, UNIT * 0.5])
        self.rect = self.canvas.create_rectangle(
            origin[0] - size_of_element, origin[1] - size_of_element,
            origin[0] + size_of_element, origin[1] + size_of_element,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)
        # return the original coordinate of the explorer in every beginning of episode so that we can
        # carry out the sequent judge action.

    def create_holes_randomly(self, number):
        origin = np.array([UNIT * 0.5, UNIT * 0.5])
        # Create origin location in the environment frame.
        holes = []
        global holes_location
        global goal_location
        hell_ = list(range(number))
        for i in range(number):
            list_y = list(range(self.height))
            list_x = list(range(self.width))
            x_location = random.choice(list_x)
            y_location = random.choice(list_y)
            # Because only 25% of the whole place is holes, there is enough room to distribute all the holes without
            # any adjacent holes. By this assume, we can avoid the situation the completed blocked situation.
            # Avoid the hole appear at the initial location & the goal location
            # Avoid the hole appear at the previous hole locations
            flag = 0
            while (
                    [x_location, y_location] == [0, 0] or
                    [x_location, y_location] in holes or
                    [x_location+1, y_location+1] in holes or
                    [x_location+1, y_location-1] in holes or
                    [x_location-1, y_location+1] in holes or
                    [x_location-1, y_location-1] in holes or
                    [x_location, y_location] in goal_location) and flag < 100:
                flag += 1
                # Avoid chose for too many times
                x_location = random.choice(list_x)
                y_location = random.choice(list_y)
            holes.append([x_location, y_location])
            hell1_center = origin + np.array([UNIT * x_location, UNIT * y_location])
            hell_append = self.canvas.create_rectangle(
                hell1_center[0] - size_of_element, hell1_center[1] - size_of_element,
                hell1_center[0] + size_of_element, hell1_center[1] + size_of_element,
                fill='blue')
            holes_location.append(self.canvas.coords(hell_append))
        return 0

    def creat_holes_directly(self, location):
        origin = np.array([UNIT * 0.5, UNIT * 0.5])
        # Create origin location in the environment frame.
        global holes_location
        for i in range(len(location)):
            x_location = location[i][0]
            y_location = location[i][1]
            hell1_center = origin + np.array([UNIT * x_location, UNIT * y_location])
            hell_append = self.canvas.create_rectangle(
                hell1_center[0] - size_of_element, hell1_center[1] - size_of_element,
                hell1_center[0] + size_of_element, hell1_center[1] + size_of_element,
                fill='blue')
            holes_location.append(self.canvas.coords(hell_append))

    def step(self, action):
        s = self.canvas.coords(self.rect)
        # move up
        if action == 0 and s[1] > UNIT:
            self.canvas.move(self.rect, 0, -UNIT)  # move agent
        # move down
        if action == 1 and s[1] < (MAZE_H - 1) * UNIT:
            self.canvas.move(self.rect, 0, + UNIT)
        # move right
        if action == 2 and s[0] < (MAZE_W - 1) * UNIT:
            self.canvas.move(self.rect, + UNIT, 0)
        # move left
        if action == 3 and s[0] > UNIT:
            self.canvas.move(self.rect, - UNIT, 0)

        state_new = self.canvas.coords(self.rect)  # next state

        # reward function 返回的值
        # judge if the explorer have find the frisbee
        if state_new in goal_location:
            reward = 1
            done = True
        # judge if the explorer have fall into the hole
        elif state_new in holes_location:
            reward = -1
            done = True
        # a normal state: the robot is still walking on the surface
        else:
            reward = 0
            done = False
        # print(f'计算前的 state_new:{state_new}')
        state_new = list([int((state_new[0]-4)/UNIT), int((state_new[1]-4)/UNIT)])
        return state_new, reward, done


