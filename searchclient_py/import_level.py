import numpy as np
from constants import *


def print_level(level):
    '''
    Print a raw level formatted as a list of strings, without \n.
    '''
    for row in range(len(level)):
        for col in range(len(level[0])):
            print(level[row][col], end='')
        print('')


def import_level(filename, printout=[0,0,0,0,0]):
    '''
    Imports .lvl text files to color dict and numpy arrays.
    If position [x,y] is a wall, then walls[x,y] is 1, otherwise 0. etc.
    INPUT:
        - filename (string): filename of level
        - printout ([string]): list of ints that specifies which level elements to print.
            e.g. [1,1,1,1] will print [raw level & colors, walls, goals, agents, boxes]
    OUTPUT: tuple with 5 elements:
                tuple[0]: colors of elements ({color:elements})
                tuple[1]: walls (numpy.ndarray)
                tuple[2]: goals (numpy.ndarray)
                tuple[3]: agents (numpy.ndarray)
                tuple[4]: boxes (numpy.ndarray)
                tuple[5]: raw map ([string])
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


    # Element types
    wall_chars = ['+']

    goal_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', \
                  'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']

    agent_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    box_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', \
                   'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

    walls = np.zeros((nrows, ncols), dtype=int)
    goals = np.zeros((nrows, ncols), dtype=int)
    agents = np.zeros((nrows, ncols), dtype=int)
    boxes = np.zeros((nrows, ncols), dtype=int)

    for i, row in enumerate(raw):
        for j, char in enumerate(row):

            if char in wall_chars:
                walls[i, j] = 1

            elif char in goal_chars:
                goals[i, j] = 1

            elif char in agent_chars:
                agents[i, j] = 1

            elif char in box_chars:
                boxes[i, j] = 1

            elif char == ' ':
                continue

            else:
                raise Exception('Invalid character in .lvl file: {}'.format(char))

    if printout[0]:
        print('ROWS: {}'.format(nrows))
        print('COLUMNS: {}'.format(ncols))
        print()
        print("COLORS:")
        for color in colors:
            print("{}: {}".format(color, colors[color]))
        print()
        print('RAW:')
        print_level(raw)
        print()

    if printout[1]:
        print('WALLS:')
        print(walls)
        print()

    if printout[2]:
        print('GOALS:')
        print(goals)
        print()

    if printout[3]:
        print('AGENTS:')
        print(agents)
        print()

    if printout[4]:
        print('BOXES:')
        print(boxes)
        print()

    return (colors, walls, goals, agents, boxes, raw)


#
# if __name__ == "__main__":
#     # test_level = 'pathfinderTest.lvl'
#     test_level = 'SAsokobanLevel96.lvl'
#     # test_level = 'MAtbsAppartment.lvl'
#
#     level = import_level(test_level, printout=[1,1,1,1,1])
#     colors, walls, goals, agents, boxes, raw = level