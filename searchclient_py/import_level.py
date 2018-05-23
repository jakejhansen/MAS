import sys
from state import Info
from state import State


def import_level(server_messages):

    try:

        line = server_messages.readline().rstrip()

        # In case of MA level, pop all first lines about colors before actual level
        colors_list = []
        while '+' not in line:  # Test if row is a color information row
            colors_list.append(line)
            line = server_messages.readline().rstrip()

        # Make dict of {colors:elements}
        colors = {}
        for color_line in colors_list:
            color_line = "".join(color_line.split())  # strip all whitespace
            color, elements = color_line.split(':')
            elements = elements.split(',')
            colors[color] = elements

        # Read in level, line by line, and detect level size
        line_save = []
        row_dim = 0
        col_dim = 0
        while line:
            line_save.append(line)  # Save current line
            row_dim += 1
            if len(line) > col_dim:  # Get max width of level (necessary if not rectangular)
                col_dim = len(line)
            line = server_messages.readline().rstrip()


        # Info contains static level data (dims, colors, walls, goals)
        # State contains dynamic level data (agents, boxes, child states etc.)
        info = Info(dims=[row_dim, col_dim])
        initial_state = State(dims=[row_dim, col_dim], info=info)

        # Write level info into "info" and "initial_state"
        if colors:
            info.colors = colors

        row = 0
        for line in line_save:
            for col, char in enumerate(line):
                if char == '+':
                    info.walls[row][col] = True
                elif char in "0123456789":
                    if initial_state.agent_row is not None:
                        print(
                            'Error, encountered a second agent (client only supports one agent).',
                            file=sys.stderr,
                            flush=True)
                        sys.exit(1)
                    initial_state.agent_row = row
                    initial_state.agent_col = col
                elif char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    initial_state.boxes[row][col] = char
                elif char in "abcdefghijklmnopqrstuvwxyz":
                    info.goals[row][col] = char
            row += 1

    except Exception as ex:
        print('Error parsing level: {}.'.format(repr(ex)), file=sys.stderr, flush=True)
        sys.exit(1)

    initial_state.make_list_representation()

    return info, initial_state
