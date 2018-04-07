import numpy as np

def import_level(filename, printmap=False):
    '''
    Imports .lvl text files to numpy arrays.
    Only imports walls right now. If position [x,y] is a wall, then walls[x,y]
    is 1, otherwise 0.
    INPUT: filename of level [string]
    OUTPUT: numpy.array [numpy.ndarray]
    '''
    LEVELS_PATH = "../levels/"
    walls = []

    if printmap == True:
        with open(LEVELS_PATH+filename, 'r') as f:
            level = f.read()
            print(level + '\n')

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


if __name__ == "__main__":
    walls = import_level('pathfinderTest.lvl', printmap=True)
    print(walls)