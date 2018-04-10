import numpy as np
from constants import *
from pathfinder import pathfinder
from pprint import pprint

def print_level(level):
    '''
    Print a raw level formatted as a list of strings, without \n.
    '''
    for row in range(len(level)):
        for col in range(len(level[0])):
            print(level[row][col], end='')
        print('')


def import_level(filename, elementtype='raw', printmap=False):
    '''
    Imports .lvl text files to numpy arrays.
    Only imports walls right now. If position [x,y] is a wall, then walls[x,y]
    is 1, otherwise 0.
    INPUT:
        - filename [string]: filename of level 
        - type [string]: type of level element to import, must be one of
            {'walls', agents, boxes, goals} (DEFAULT: 'walls')
        - printmap: If true, prints the whole map before import and after import
            [bool] (DEFAULT: 'False')
    OUTPUT: numpy.array [numpy.ndarray]
    '''

    with open(LEVELS_PATH + filename, 'r') as f:
        raw = f.readlines()

        # Remove \n at end of every line
        for i, line in enumerate(raw):
            raw[i] = line[:-1]

        # Pop all lines about colors before level
        colors_raw = []
        while '+' not in raw[0]:
            colors_raw.append(raw.pop(0))

        # Make dict of {colors:elements}
        colors = {}
        for line in colors_raw:
            line = "".join(line.split()) # strip all whitespace
            color, elements = line.split(':')
            elements = elements.split(',')
            colors[color] = elements

        # Determine number of rows (nrows) and longest row length (ncols) of level
        nrows = len(raw)
        ncols = 0
        for row in raw:
            if len(row) > ncols:
                ncols = len(row)

        # Pad all lines with spaces until longest row length (ncols) for rect level
        for i, row in enumerate(raw):
            raw[i] = row.ljust(ncols)

        # Convert rows from strings to lists of chars
        for i, line in enumerate(raw):
            raw[i] = list(line)

        raw = np.array(raw)


    if elementtype == 'raw':
        if printmap:
            print_level(raw)
        return (colors, raw)

    elif elementtype == 'walls':
        keys = ['+']

    elif elementtype == 'agents':
        keys = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    elif elementtype == 'boxes':
        keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', \
            'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

    elif elementtype == 'goals':
        keys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', \
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']

    else:
        raise Exception('Invalid type parameter')

    if elementtype != 'raw':
        level = np.zeros((nrows,ncols), dtype=int)

        for i, row in enumerate(raw):
            for j, char in enumerate(row):
                if char in keys:
                    level[i,j] = 1

        if printmap:
            print(level)

        return (colors, level)


if __name__ == "__main__":
    # test_level = 'pathfinderTest.lvl'
    # test_level = 'SAsokobanLevel96.lvl'
    test_level = 'MAtbsAppartment.lvl'

    print('RAW:')
    raw = import_level(test_level, elementtype='raw', printmap=True)
    print()

    print('WALLS:')
    walls = import_level(test_level, elementtype='walls', printmap=True)
    print()
    
    print('GOALS:')
    goals = import_level(test_level, elementtype='goals', printmap=True)

    print('BOXES:')
    goals = import_level(test_level, elementtype='boxes', printmap=True)










