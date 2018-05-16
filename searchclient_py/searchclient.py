"""
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
"""

import argparse
import re
import sys
import time
from time import gmtime, strftime

from tabulate import tabulate

import memory
import shutil
from state import State
from state import Info
from strategy import StrategyBFS, StrategyDFS, StrategyBestFirst, Custom
from heuristic import AStar, WAStar, Greedy


class SearchClient:
    def __init__(self, server_messages, init_state=None, desired_agent_pos=None):

        if desired_agent_pos is None:
            desired_agent_pos = [None, None]
        if server_messages is not None:
            self.initial_state = None

            try:

                line = server_messages.readline().rstrip()

                # Pop all lines about colors before level
                colors_list = []
                while '+' not in line: # Test if row is a color information row
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

                # Write level info into initial state
                self.info = Info(dims=[row_dim, col_dim])
                self.initial_state = State(dims=[row_dim, col_dim], info=self.info)

                if colors:
                    self.info.colors = colors

                row = 0
                for line in line_save:
                    for col, char in enumerate(line):
                        if char == '+':
                            self.info.walls[row][col] = True
                        elif char in "0123456789":
                            if self.initial_state.agent_row is not None:
                                print(
                                    'Error, encountered a second agent (client only supports one agent).',
                                    file=sys.stderr,
                                    flush=True)
                                sys.exit(1)
                            self.initial_state.agent_row = row
                            self.initial_state.agent_col = col
                        elif char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                            self.initial_state.boxes[row][col] = char
                        elif char in "abcdefghijklmnopqrstuvwxyz":
                            self.info.goals[row][col] = char
                    row += 1

            except Exception as ex:
                print('Error parsing level: {}.'.format(repr(ex)), file=sys.stderr, flush=True)
                sys.exit(1)

            self.initial_state.make_list_representation()

        else:
            self.initial_state = init_state
            self.info = Info(dims=[init_state.MAX_ROW, init_state.MAX_COL], agent=desired_agent_pos)

    def search(self, strategy) -> '[State, ...]':
        print('Starting search with strategy {}.'.format(strategy), file=sys.stderr, flush=True)
        strategy.add_to_frontier(self.initial_state)

        iterations = 0
        while True:
            if iterations == 1000:
                print(strategy.search_status(), file=sys.stderr, flush=True)
                iterations = 0

            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None

            if strategy.frontier_empty():
                return None

            leaf = strategy.get_and_remove_leaf()

            if leaf.is_goal_state():
                return leaf.extract_plan()

            strategy.add_to_explored(leaf)
            for child_state in leaf.get_children():
                if not strategy.is_explored(child_state) and not strategy.in_frontier(child_state):
                    strategy.add_to_frontier(child_state)

            iterations += 1

    def search2(self, strategy, goalstate) -> '[State, ...]':
        print('Starting search with strategy {}.'.format(strategy), file=sys.stderr, flush=True)
        strategy.add_to_frontier(self.initial_state)

        iterations = 0
        while True:

            # if iterations >= 1:
            #     print("\033[H\033[J") #Stack overflow to clear screen
            #     print(leaf) #Print state
            #     input() #Wait for user input

            if iterations >= 1000:
                print(strategy.search_status(), file=sys.stderr, flush=True)
                iterations = 0

            if memory.get_usage() > memory.max_usage:
                print('Maximum memory usage exceeded.', file=sys.stderr, flush=True)
                return None

            if strategy.frontier_empty():
                return None

            leaf = strategy.get_and_remove_leaf()

            if leaf.is_goal_state2(goalstate):
                return leaf.extract_plan(), leaf

            strategy.add_to_explored(leaf)
            for child_state in leaf.get_children():
                if not strategy.is_explored(child_state) and not strategy.in_frontier(child_state):
                    strategy.add_to_frontier(child_state, goalstate)

            iterations += 1


def main(strat, lvl, log):
    log_name = strftime("%Y-%m-%d-%H-%M", gmtime())
    start = time.time()

    # Read server messages from stdin.
    if lvl == "":
        server_messages = sys.stdin
    else:
        server_messages = open("../levels/" + lvl, "r")

    # Use stderr to print to console through server.
    print('SearchClient initializing. I am sending this using the error output stream.',
          file=sys.stderr, flush=True)
    
    # Read level and create the initial state of the problem.
    client = SearchClient(server_messages)

    # c2 = SearchClient(server_messages=None, init_state=client.initial_state)

    if strat == "BFS":
        strategy = StrategyBFS()
    # Ex. 1:
    elif strat == "DFS":
        strategy = StrategyDFS()

    # Ex. 3:

    elif strat == "astar":
        strategy = StrategyBestFirst(AStar(client.initial_state))
    elif strat == "wstar":
        strategy = StrategyBestFirst(WAStar(client.initial_state,
                                            5))  # You can test other W values than 5, but use 5 for the report.
    elif strat == "greedy":
        strategy = StrategyBestFirst(Greedy(client.initial_state))

    elif strat == "custom":
        strategy = Custom(client.initial_state)
    else:
        raise Exception("Invalid strategy")

    if strat != "custom":
        solution = client.search(strategy)

    else:
        solution = strategy.return_solution()

    if solution is None:
        print(strategy.search_status(), file=sys.stderr, flush=True)
        print('Unable to solve level.', file=sys.stderr, flush=True)
        sys.exit(0)
    elif not log:
        print('\nSummary for {}.'.format(strategy), file=sys.stderr, flush=True)
        print('Found solution of length {}.'.format(len(solution)), file=sys.stderr, flush=True)
        # print('{}.'.format(strategy.search_status()), file=sys.stderr, flush=True)

        for state in solution:
            print(state.action, flush=True)
            response = server_messages.readline().rstrip()
            if response == 'false':
                print('Server responded with "{}" to the action "{}" applied in:\n{}\n'.format(
                    response, state.action, state), file=sys.stderr, flush=True)
                break
    else:
        # Log info
        print('Found solution of length {}.'.format(len(solution)), file=sys.stderr, flush=True)
        with open("logs/" + log_name, "a") as myfile:
            myfile.write(tabulate([[lvl.ljust(22), format(len(solution)),
                                    "{0:.2f}".format(time.time() - start)]],
                                  tablefmt="plain") + "\n")
        shutil.copy2('strategy.py', "logs/" + log_name + "_strategy")


if __name__ == '__main__':
    # Program arguments.
    parser = argparse.ArgumentParser(description='Simple client based on state-space graph search.')
    parser.add_argument('--max_memory', metavar='<MB>', type=float, default=512.0,
                        help='The maximum memory usage allowed in MB (soft limit, default 512).')
    parser.add_argument('--strategy', default="BFS",
                        help='The chosen strategy BFS | DFS | ASTAR | WASTAR | GREEDY')
    parser.add_argument('--lvl', default="",
                        help="Choose to input lvl, mode made when running without server")
    parser.add_argument('--log', default=False, help="Log to file")
    args = parser.parse_args()

    # Set max memory usage allowed (soft limit).
    memory.max_usage = args.max_memory
    my_strat = args.strategy

    # Run client.
    main(my_strat, args.lvl, args.log)
