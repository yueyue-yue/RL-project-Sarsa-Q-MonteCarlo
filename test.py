import numpy as np
import random

map = np.zeros(100).reshape(10, 10)

have_reached = []
store = []
count = 1

def dfs(location):
    if location in have_reached:
        return
    have_reached.append(location)
    up = [location[0], location[1] - 1]
    right = [location[0] + 1, location[1]]
    left = [location[0] - 1, location[1]]
    down = [location[0], location[1] + 1]
    if (0 <= up[0] <= 9) and (0 <= up[1] <= 9):
        if map[location[0], location[1] - 1] != 1:
            map[location[0], location[1] - 1] = 99
            dfs(up)
    if (0 <= right[0] <= 9) and (0 <= right[1] <= 9):
        if map[location[0] + 1, location[1]] != 1:
            map[location[0] + 1, location[1]] = 99
            dfs(right)
    if (0 <= left[0] <= 9) and (0 <= left[1] <= 9):
        if map[location[0] - 1, location[1]] != 1:
            map[location[0] - 1, location[1]] = 99
            dfs(left)
    if (0 <= down[0] <= 9) and (0 <= down[1] <= 9):
        if map[location[0], location[1] + 1] != 1:
            map[location[0], location[1] + 1] = 99
            dfs(down)
    return


def create_holes_randomly(number):
    # Create origin location in the environment frame.
    holes = []
    for i in range(number):
        list_y = list(range(10))
        list_x = list(range(10))
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
                [x_location, y_location] == [7, 2]):
            flag += 1
            # Avoid chose for too many times
            x_location = random.choice(list_x)
            y_location = random.choice(list_y)
        holes.append([x_location, y_location])
        print('holes', holes)
    return holes


holes = create_holes_randomly(25)
for i in holes:
    map[i[0], i[1]] = 1
for i in range(10):
    map[i, 2] = 1
print(map)
dfs([0, 0])
print(map)
count = 0
for i in range(10):
    for j in range(10):
        if map[i, j] == 99:
            count += 1
print('count:', count)
'''
print(map[0, 1])
if map[0, 1] != 1:
    print('yes')
'''
