import numpy as np
from constants import *
from import_level import *
from pathfinder import pathfinder

def corner_finder(filename, print_corners=False):
    '''Find all corners of a level.
    A corner is defined: cell c with the property that if there is a path from
    A to B (with A ≠ c and B ≠ c) in the level, then there is a path that
    doesn’t go through c.
    Or as implemented here:
    1. Corner-candidates: have two touching walls as neighbors
    2. Corners: corner-candidates where there is a path between two non-wall
    neighbors that doens't go through c.
        INPUT:
            - filename: filename of level [string]
            - print_corners: if true, then prints level for X for corners
        OUTPUT: tuple of list of coordinates: (corners,corner_goals)
    '''
    print("HEJ1")
    _, walls, goals, _, _, raw = import_level(filename)
    print("HEJ2")

    nrows, ncols = walls.shape

    corners = []
    corner_goals = []
    # we skip first/last row/col (these are either walls or not part of level)
    for i in range(1, nrows-1):
        for j in range(1, ncols-1):

            # Pass if current cell is a wall
            if walls[i,j] == 1:
                continue
            else:
                neighbor_N = neighbor_S = neighbor_E = neighbor_W = 0
                # Neighbor coordinates
                N = (i-1, j)
                S = (i+1, j)
                E = (i, j+1)
                W = (i, j-1)
                # Neighbors are either 0 (free) or 1 (walls)
                neighbor_N = walls[N]
                neighbor_S = walls[S]
                neighbor_W = walls[W]
                neighbor_E = walls[E]

                # if three neighbors are walls, then (i,j) is corner point (and a
                # "blind-alley point")
                if neighbor_N + neighbor_S + neighbor_E + neighbor_W == 3:
                    if goals[i,j] == 1:
                        corner_goals.append((i,j))
                    else:
                        corners.append((i,j))

                # if two neighbors are touching, see if it's a corner
                elif  (neighbor_W + neighbor_N) == 2:
                        # try to find path between S and E neighbors
                        walls[i,j] = 1  # add temp wall at corner candidate
                        if pathfinder(walls, S, E):
                            if goals[i,j] == 1:
                                corner_goals.append((i,j))
                            else:
                                corners.append((i,j))
                        walls[i,j] = 0  # remove temp wall at corner candidate

                elif  (neighbor_N + neighbor_E) == 2:
                        # try to find path between S and W neighbors
                        walls[i,j] = 1  # add temp wall at corner candidate
                        if pathfinder(walls, S, W):
                            if goals[i,j] == 1:
                                corner_goals.append((i,j))
                            else:
                                corners.append((i,j))
                        walls[i,j] = 0  # remove temp wall at corner candidate

                elif  (neighbor_S + neighbor_E) == 2:
                        # try to find path between N and W neighbors
                        walls[i,j] = 1  # add temp wall at corner candidate
                        if pathfinder(walls, N, W):
                            if goals[i,j] == 1:
                                corner_goals.append((i,j))
                            else:
                                corners.append((i,j))
                        walls[i,j] = 0  # remove temp wall at corner candidate

                elif  (neighbor_S + neighbor_W) == 2:
                    # try to find path between N and E neighbors
                        walls[i,j] = 1  # add temp wall at corner candidate
                        if pathfinder(walls, N, E):
                            if goals[i,j] == 1:
                                corner_goals.append((i,j))
                            else:
                                corners.append((i,j))
                        walls[i,j] = 0  # remove temp wall at corner candidate

    if print_corners:
        print('ROWS: {}'.format(nrows))
        print('COLS: {} \n'.format(ncols))

        for point in corners:
            x, y = point
            raw[x][y] = '$'
        for point in corner_goals:
            x, y = point
            raw[x][y] = '€'
        print()
        print("WITH CORNERS:\n")
        print_level(raw)
        print()
        print("Corners:")
        print(corners)
        print()
        print("Corner-goals:")
        print(corner_goals)
        print()
    return (corners, corner_goals)



if __name__ == "__main__":
    print("HEJ0")
    # corner_finder('pathfinderTest.lvl', print_corners=True)
    # corner_finder('SAlabyrinth.lvl', print_corners=True)
    # corner_finder('SAsimple3.lvl', print_corners=True)
    # corner_finder('SAsokobanLevel96.lvl', print_corners=True)
    test = corner_finder('SAsokobanLevel96.lvl', print_corners=True)












