import numpy as np

def import_level(filename):
    '''
    Imports .lvl text files to numpy arrays.
    Only imports walls right now. If position [x,y] is a wall, then walls[x,y]
    is 1, otherwise 0.
    INPUT: filename of level [string]
    OUTPUT: numpy.array [numpy.ndarray]

    '''
    walls = []
    with open(LEVELS_PATH+filename, 'r') as f:
        rows = 0
        for line in f:
            cols = len(line) # give correct result on last iteration
            rows += 1

            wall_line = []
            for char in line:
                if char == '\n':
                    continue
                elif char == '+':
                    wall_line = np.append(wall_line, '1')
                else:
                    wall_line = np.append(wall_line, '0')

            wall_line = np.array(wall_line)
            walls.append(wall_line)

        walls = np.array(walls)
        return walls



# Author: Christian Careaga (christian.careaga7@gmail.com)
# A* Pathfinding in Python (2.7)
# Please give credit if used
# https://github.com/ActiveState/code/blob/master/recipes/Python/578919_PythPathfinding_Binary/recipe-578919.py


import numpy
from heapq import *


def pathfinder(array, start, goal):
    def heuristic(a, b):
        return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

    # neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.reverse()
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
                
    return False

'''Here is an example of using my algo with a numpy array,
   astar(array, start, destination)
   astar function returns a list of points (shortest path)'''
if __name__ == "__main__":
    nmap = numpy.array([
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    # x = astar(nmap, (0,0), (10,13))
    # 5*(11 + 2) + 2 = 67 as expected

    def path_and_length(array, start, goal):
        path = astar(array, start, goal)
        return (len(path), path)

    x = path_and_length(nmap, (0,0), (10,13))


    rows, cols = nmap.shape



    paths = numpy.empty([rows, cols], dtype=np.ndarray)

    start = (0,0)
    paths_from_start = numpy.empty([rows, cols], dtype=tuple)
    for i in range(rows):
        for j in range(cols):
            # don't calculate path from start to start (distance is 0)
            if start == (i, j):
                continue
            # don't calculate path from start to a wall
            elif nmap[i,j] == 1:
                continue
            else:
                # re-use path if already calculated from (i,j) to start
                if 0 != 0:
                    pass # TODO re-use calculation here!
                else:
                    # calculate path from start to (i,j)
                    paths_from_start[i,j] = path_and_length(nmap, start, (i, j))

