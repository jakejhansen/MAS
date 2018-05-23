import numpy as np

from import_level import *
from pathfinder import pathfinder


def corner_finder(walls, goals, print_corners=False):
    """Find all corners of a level.
    A corner is defined: cell c with the property that if there is a path from
    A to B (with A ≠ c and B ≠ c) in the level, then there is a path that
    doesn't go through c.
    Or as implemented here:
    1. Corner-candidates: have two diagonally touching walls as neighbors
    2. Corners: corner-candidates where there is a path between two non-wall
    neighbors that doesn't go through c.
        INPUT:
            - filename: filename of level [string]
            - print_corners: if true, then prints level for X for corners
        OUTPUT: tuple of list of coordinates: (corners,corner_goals)
    """

    raw = np.zeros_like(walls, dtype='object')

    nrows, ncols = walls.shape

    corners = []
    corner_goals = []
    # we skip first/last row/col (these are either walls or not part of level)
    for i in range(1, nrows - 1):
        for j in range(1, ncols - 1):

            if is_corner(walls, [i, j]):
                if goals[i, j] == 1:
                    corner_goals.append((i, j))
                else:
                    corners.append((i, j))

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
    return corners, corner_goals


def is_corner(walls, position) -> bool:
    """ Return True if position is a corner, False otherwise."""
    i, j = position
    is_corner = False

    # Special case #1: Always false if cell is a wall
    if walls[i, j] == 1:
        return False

    # Neighbor relative coordinates
    n = (i - 1, j)
    s = (i + 1, j)
    e = (i, j + 1)
    w = (i, j - 1)
    # Neighbors are either 0 (non-wall) or 1 (walls)
    neighbor_n = walls[n]
    neighbor_s = walls[s]
    neighbor_w = walls[w]
    neighbor_e = walls[e]

    # Special case #2: Always true if three neighbors are walls
    if neighbor_n + neighbor_s + neighbor_e + neighbor_w == 3:
        return True

    # add temp wall at corner candidate, try to find path between neighbors pairwise
    walls[i,j] = 1

    if (neighbor_w + neighbor_n) == 2:
        if pathfinder(walls, s, e):
            is_corner = True

    elif (neighbor_n + neighbor_e) == 2:
        if pathfinder(walls, s, w):
            is_corner = True

    elif (neighbor_s + neighbor_e) == 2:
        if pathfinder(walls, n, w):
            is_corner = True

    elif (neighbor_s + neighbor_w) == 2:
        if pathfinder(walls, n, e):
            is_corner = True

    walls[i,j] = 0

    return is_corner


if __name__ == "__main__":
    # filename = 'SAlabyrinth.lvl'
    filename = 'SAsimple3.lvl'
    # filename = 'SAsokobanLevel96.lvl'# import_level()

    _, test_walls, test_goals, _, _, test_raw = import_level(filename,
                                                             printout=[1, 0, 0, 0, 0])

    test = corner_finder(test_walls, test_goals, print_corners=True)
