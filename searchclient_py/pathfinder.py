from heapq import *

import numpy as np


# Author: Christian Careaga (christian.careaga7@gmail.com)
# A* Pathfinding in Python (2.7)
# Please give credit if used
# https://github.com/ActiveState/code/blob/master/recipes/Python/578919_PythPathfinding_Binary/recipe-578919.py

# EDIT: Updated by Gandalf Saxe
# Made compatile with Python 3 (just a print -> print())

def heuristic(a, b):
    return np.abs(b[0] - a[0]) + np.abs(b[1] - a[1])

def pathfinder(array, start, goal):
    '''
    Find shortest path between `start` and `goal` using A* search.
    '''

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


def path_and_length(array, start, goal):
    path = pathfinder(array, start, goal)
    return (len(path), path)


if __name__ == "__main__":

    # Testing pathfinder()

    nmap = np.array([
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

    print(pathfinder(nmap, (0,0), (10,13)))
    print()
    # 5*(11 + 2) + 2 = 67 as expected

    # Testing path_and_length
    print(path_and_length(nmap, (0,0), (10,13)))






# Below is unfinished / unused code for pre-processing an all-paths matrix.
# ------------------------------------------------------------------------------

# rows, cols = nmap.shape

# paths = numpy.zeros([rows, cols], dtype=np.ndarray)

# t0 = time.time()

# for x in range(rows):
#     for y in range(cols):
#         start = (x,y)
#         if nmap[x,y] == 1:
#             continue
#         paths_from_start = numpy.zeros([rows, cols], dtype=tuple)
#         for i in range(rows):
#             for j in range(cols):
#                 goal = (i, j)
#                 # don't calculate path from start to start (distance is 0)
#                 if start == goal:
#                     continue
#                 # don't calculate path from start to a wall
#                 elif nmap[goal] == 1:
#                     continue
#                 # get shortest path from start to goal
#                 else:
#                     # if paths[goal] == 0:
#                     if np.array_equal(paths[goal], 0):
#                         paths_from_start[goal] = path_and_length(nmap, start, goal)
#                     # re-use path if already calculated from goal to start
#                     else:
#                         paths_from_start[goal] = paths[goal][start]

#         paths[start] = paths_from_start
# t1 = time.time()

# total = t1-t0
# print(total)

# # Testing that path is the same both ways

# test1 = paths[0,0][10,13]
# trim1 = np.delete(test1[1],0,0)
# trim1 = np.delete(test1[1],-1,0)
# test1 = (test1[0], trim1)

# test2 = paths[10,13][0,0]
# trim2 = np.delete(test2[1],0,0)
# trim2 = np.delete(test2[1],-1,0)
# trim2 = np.flip(trim2, axis=0)
# test2 = (test2[0], trim2)

# numpy.testing.assert_equal(test1, test2)