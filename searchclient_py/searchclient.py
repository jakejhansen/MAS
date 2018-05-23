"""
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
"""

import argparse
import os
import shutil
import sys
import time
from time import localtime, strftime

from tabulate import tabulate

import memory
from heuristic import AStar, WAStar, Greedy
from import_level import import_level
from planner import Custom
from state import Info
from strategy import StrategyBFS, StrategyDFS, StrategyBestFirst


class SearchClient:
    def __init__(self, server_messages, init_state=None, desired_agent_pos=None):
        if desired_agent_pos is None:
            desired_agent_pos = [None, None]

        # If level not yet imported, then import level
        if server_messages is not None:
            self.info, self.initial_state = import_level(server_messages)
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

    def search2(self, strategy, goalstate, display=False, msg="", max_time = 300) -> '[State, ...]':
        start = time.perf_counter()

        if msg == "":
            print('Starting search with strategy {}.'.format(strategy), file=sys.stderr, flush=True)

        else:
            search_method = strategy.__repr__()[strategy.__repr__().find("using"):]
            print('Starting search for: ' + msg + " | " + search_method, file=sys.stderr,
                  flush=True)

        strategy.add_to_frontier(self.initial_state)

        iterations = 0
        while True:

            if display:
                if iterations >= 1:
                    print("\033[H\033[J")  # Stack overflow to clear screen
                    print(leaf)  # Print state
                    input()  # Wait for user input

            if iterations >= 1000:
                print(strategy.search_status(), file=sys.stderr, flush=True)
                iterations = 0

            if iterations % 100 == 0:
                if (time.perf_counter() - start) > max_time:
                    return None, None

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


    def search3(self, strategy, goalstate) -> '[State, ...]':
        print('Starting search with strategy {}.'.format(strategy), file=sys.stderr,
              flush=True)
        strategy.add_to_frontier(self.initial_state)

        iterations = 0
        while True:

            if iterations >= 1:
                print("\033[H\033[J")  # Stack overflow to clear screen
                print(leaf)  # Print state
                input()  # Wait for user input

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
                if not strategy.is_explored(child_state) and not strategy.in_frontier(
                        child_state):
                    strategy.add_to_frontier(child_state, goalstate)

            iterations += 1


def main(strat, lvl, log):
    log_name = strftime("%Y-%m-%d-%H-%M", localtime())
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

        directory = "logs/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        print('Found solution of length {}.'.format(len(solution)), file=sys.stderr, flush=True)
        with open(directory + log_name, "a") as myfile:
            myfile.write(tabulate([[lvl.ljust(22), format(len(solution)),
                                    "{0:.2f}".format(time.time() - start)]],
                                  tablefmt="plain") + "\n")
        shutil.copy2('strategy.py', "logs/" + log_name + "_strategy")


if __name__ == '__main__':
    # Program arguments.
    parser = argparse.ArgumentParser(description='Simple client based on state-space graph search.')
    parser.add_argument('--max_memory', metavar='<MB>', type=float, default=3500.0,
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
