'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


import argparse
import re
import sys

import memory
from state import State
from state import Info
from strategy import StrategyBFS, StrategyDFS, StrategyBestFirst
from heuristic import AStar, WAStar, Greedy


class SearchClient:
    def __init__(self, server_messages, init_state = None, desired_agent_pos = [None, None]):

        if server_messages is not None:
            self.initial_state = None

            colors_re = re.compile(r'^[a-z]+:\s*[0-9A-Z](\s*,\s*[0-9A-Z])*\s*$')
            try:
                # Read lines for colors. There should be none of these in warmup levels.
                line = server_messages.readline().rstrip()
                if colors_re.fullmatch(line) is not None:
                    print('Invalid level (client does not support colors).', file=sys.stderr, flush=True)
                    sys.exit(1)

                line_save = []

                row_dim = 0
                col_dim = len(line)
                while line:
                    line_save.append(line) #Save current line
                    row_dim += 1
                    line = server_messages.readline().rstrip()


                # Read lines for level.
                self.info = Info(dims = [row_dim, col_dim])
                self.initial_state = State(dims = [row_dim, col_dim], info = self.info)

                row = 0
                for line in line_save:
                    for col, char in enumerate(line):
                        if char == '+':
                            self.info.walls[row][col] = True
                        elif char in "0123456789":
                            if self.initial_state.agent_row is not None:
                                print('Error, encoutered a second agent (client only supports one agent).', file=sys.stderr, flush=True)
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

        else:
            self.initial_state = init_state
            self.info = Info(dims=[init_state.MAX_ROW, init_state.MAX_COL], agent=desired_agent_pos)
            self.info
            import IPython
            IPython.embed()

    def search(self, strategy: 'Strategy') -> '[State, ...]':
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
            
            if leaf.is_goal_state() or 2 == 3:
                return leaf.extract_plan()
            
            strategy.add_to_explored(leaf)
            for child_state in leaf.get_children():
                if not strategy.is_explored(child_state) and not strategy.in_frontier(child_state):
                    strategy.add_to_frontier(child_state)
            
            iterations += 1


def main(strat, lvl):
    # Read server messages from stdin.
    if lvl == "":
        server_messages = sys.stdin
    else:
        server_messages = open("../levels/" + lvl, "r")


    # Use stderr to print to console through server.
    print('SearchClient initializing. I am sending this using the error output stream.', file=sys.stderr, flush=True)
    
    # Read level and create the initial state of the problem.
    client = SearchClient(server_messages)

    c2 = SearchClient(server_messages=None, init_state=client.initial_state)
    
    if strat == "BFS":
        strategy = StrategyBFS()
    # Ex. 1:
    elif strat == "DFS":
        strategy = StrategyDFS()
    
    # Ex. 3:

    elif strat == "astar":
        strategy = StrategyBestFirst(AStar(client.initial_state))
    elif strat == "wstar":
        strategy = StrategyBestFirst(WAStar(client.initial_state, 5)) # You can test other W values than 5, but use 5 for the report.
    elif strat == "greedy":
        strategy = StrategyBestFirst(Greedy(client.initial_state))
    
    solution = client.search(strategy)
    if solution is None:
        print(strategy.search_status(), file=sys.stderr, flush=True)
        print('Unable to solve level.', file=sys.stderr, flush=True)
        sys.exit(0)
    else:
        print('\nSummary for {}.'.format(strategy), file=sys.stderr, flush=True)
        print('Found solution of length {}.'.format(len(solution)), file=sys.stderr, flush=True)
        print('{}.'.format(strategy.search_status()), file=sys.stderr, flush=True)
        
        for state in solution:
            print(state.action, flush=True)
            response = server_messages.readline().rstrip()
            if response == 'false':
                print('Server responsed with "{}" to the action "{}" applied in:\n{}\n'.format(response, state.action, state), file=sys.stderr, flush=True)
                break

    import IPython
    IPython.embed()

if __name__ == '__main__':
    # Program arguments.
    parser = argparse.ArgumentParser(description='Simple client based on state-space graph search.')
    parser.add_argument('--max_memory', metavar='<MB>', type=float, default=512.0, help='The maximum memory usage allowed in MB (soft limit, default 512).')
    parser.add_argument('--strategy', default = "BFS", help='The chosen strategy BFS | DFS | ASTAR | WASTAR | GREEDY')
    parser.add_argument('--lvl', default = "", help="Choose to input lvl, mode made when running without server")
    args = parser.parse_args()
    
    # Set max memory usage allowed (soft limit).
    memory.max_usage = args.max_memory
    strat = args.strategy

    # Run client.
    main(strat, args.lvl)



