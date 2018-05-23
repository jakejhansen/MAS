from abc import ABCMeta, abstractmethod
from collections import deque, defaultdict
from time import perf_counter
from pathfinder import pathfinder
from corner_finder import corner_finder

import memory
import heapq
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy

class Custom():
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
        self.corner_list = corner_finder(self.state.walls.astype('int'), self.state.goals)[0]

        self.solution = self.initStrategy()

        # self.solution = self.solve_subgoals()

        # self.solution = self.return_solution2()

    def initStrategy(self):
        subgoals = []
        goals = self.state.goal_list
        boxes = self.state.box_list
        boxes = boxes.tolist()

        #Make graph and find edges
        G, labels = self.construct_graph()
        completed_goals = []
        completed_goals_index = []
        G, completed_goals, completed_goals_index = \
            self.compute_goalgraph_edges(self.state, G, labels, completed_goals, completed_goals_index, boxes)


        taken = []
        subgoals = []
        tot_solution = []

        #Start to solve the subgoals
        for i in completed_goals_index:
            goal = self.state.goal_list[i]
            box_id = self.find_best_box(goal, boxes, taken)
            box = self.state.box_list.tolist()[box_id]

            # Move agent to box, imaginarily.
            subgoals.append(self.get_adjacent_box_loc([box[0], box[1]]))
            solution, state_imaginary = self.move_agt_next_to_box(deepcopy(self.state),
                                                                  box,
                                                                  subgoals)
            # tot_solution.append(solution)

            if not self.path_is_clear(state_imaginary,
                                      [state_imaginary.agent_row, state_imaginary.agent_col],
                                      goal,
                                      box_ignore=box):

                # Find path for moving box to goal with no other boxes
                _, _, path, path_order = self.find_path_with_blocking(goal,
                                                                      box,
                                                                      deepcopy(self.state),
                                                                      subgoals)

                blocking_boxes = self.find_blocking_path(path_order,
                                                         self.state.box_list.tolist(),
                                                         ignore_box=self.state.box_list.tolist()[
                                                             box_id])

                # Find out where to place the blocking boxes if there is any:
                if len(blocking_boxes) > 0:
                    pos_blocking_boxes = []  # Find where to place these boxes
                    agent_positions = []
                    for block_box in blocking_boxes[::-1]:
                        pos, path, path_order, agent_pos = self.find_pos_blocks(block_box,
                                                                                blocking_boxes,
                                                                                path,
                                                                                path_order,
                                                                                deepcopy(
                                                                                    self.state))

                        pos_blocking_boxes.append([block_box, pos.tolist()])
                        agent_positions.append(agent_pos)

                    # Move agent to first box
                    bb = self.state.box_list[blocking_boxes[0]]
                    subgoals.append(self.get_adjacent_box_loc([bb[0], bb[1]]))
                    solution, self.state = self.move_agt_next_to_box(self.state,
                                                                     bb,
                                                                     subgoals)
                    tot_solution.append(solution)

                    # Move the blocking boxes to their respective positions
                    plan_move_box, self.state = self.move_blocking_boxes(pos_blocking_boxes,
                                                                         path,
                                                                         deepcopy(self.state))

                    tot_solution.append(plan_move_box)

            # Go back to box
            box = self.state.box_list.tolist()[box_id]
            solution, self.state = self.move_agt_next_to_box(self.state,
                                                             box,
                                                             subgoals)
            tot_solution.append(solution)

            subgoals.append([box_id, goal[:2]])
            # Move box to goal
            import searchclient
            import strategy
            import heuristic

            client = searchclient.SearchClient(server_messages=None,
                                               init_state=deepcopy(self.state))
            strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
            # Set other to walls
            init_backup = deepcopy(client.initial_state)
            init_backup.parent = None
            client.initial_state.walls[client.initial_state.boxes != None] = 1
            client.initial_state.walls[box[0]][box[1]] = 0
            client.initial_state.boxes[client.initial_state.boxes != None] = None
            client.initial_state.boxes[box[0]][box[1]] = box[2].upper()
            client.initial_state.box_list = np.array([box], dtype="object")
            if self.path_is_clear(client.initial_state,
                                  [box[0], box[1]],
                                  goal[:2],
                                  box_ignore=box):
                solution, temp_state = client.search2(strategy, [[0, goal[:2]]],
                                                      msg="Box {} to Goal {} - No other "
                                                          "boxes".format(goal,
                                                          box))

                # Remove old pos
                init_backup.boxes[init_backup.box_list[box_id][0]][
                    init_backup.box_list[box_id][1]] = None
                # Put at new pos
                init_backup.boxes[temp_state.box_list[0][0]][temp_state.box_list[0][1]] = \
                    temp_state.box_list[0][
                        2].upper()

                init_backup.box_list[box_id] = temp_state.box_list[0]
                init_backup.agent_col = temp_state.agent_col
                init_backup.agent_row = temp_state.agent_row
                temp_state = init_backup

            # Else do the normal thingie and hope it works
            else:
                import searchclient
                import strategy
                import heuristic
                client = searchclient.SearchClient(server_messages=None,
                                                   init_state=deepcopy(self.state))
                strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
                solution, temp_state = client.search2(strategy, subgoals,
                                                      msg = "Box {} to Goal{} - With other "
                                                            "boxes".format(goal, box))
                temp_state.parent = None

            tot_solution.append(solution)
            self.state = temp_state

            taken.append(box_id)

        return tot_solution


    def compute_goalgraph_edges(self, state, G, labels, completed_goals, completed_goals_index,
                                boxes):
        counter = 0
        while len(completed_goals) != len(G.nodes):
            for i in G:
                if i not in completed_goals_index:
                    goal = state.goal_list[i]
                    # Check if completing the goal will block the completion of other goals
                    nmap = state.walls.astype('int')
                    nmap[goal[0]][
                        goal[1]] = 1  # Pretend that current goal is occupied -> make into wall
                    for g in completed_goals:
                        nmap[g[0]][g[1]] = 1
                    all_goals_completeable = True
                    for j, other_goal in enumerate(state.goal_list):
                        if j != i and j not in completed_goals_index:
                            goal_completable = False
                            for b_n, box in enumerate(boxes):
                                if box[2] == other_goal[2]:  # same type
                                    v = pathfinder(nmap, (box[0], box[1]),
                                                   (other_goal[0], other_goal[1]))
                                    if v:
                                        goal_completable = True  # goal achievable by *some* box of same type

                            if goal_completable == False:
                                G.add_edge(j, i)
                                all_goals_completeable = False

                    if all_goals_completeable:
                        completed_goals.append(goal)
                        completed_goals_index.append(i)

            counter += 1
            if counter > 100:
                completed_goals_index, labels = self.topological_sort_with_cycles(G, labels)
                break

        return G, completed_goals, completed_goals_index

    def path_is_clear(self, state, start, finish, box_ignore=None):
        nmap = state.walls.astype('int')
        nmap[state.boxes != None] = 1
        if box_ignore:
            nmap[box_ignore[0]][box_ignore[1]] = 0

        v = pathfinder(nmap, (start[0], start[1]),
                       (finish[0], finish[1]))
        if v:
            return True

        return False

    def move_agt_next_to_box(self, state, box, subgoals):
        """
        Moves the agent next to a given box [row, col, type]
        Args:
            state: current state
            box: target box [row, col, type]

        Returns:
            state: new state
            solution: the solution for getting to the new state
        """
        import searchclient
        import strategy
        import heuristic
        subgoals.append(self.get_adjacent_box_loc([box[0], box[1]]))
        client = searchclient.SearchClient(server_messages=None, init_state=state)
        client.initial_state.desired_agent = subgoals[-1]  # Last subgoal is to move the agent
        strategy = strategy.StrategyBestFirst(heuristic.Greedy(client.initial_state))
        solution, state = client.search2(strategy, subgoals, msg = "Agent to box {}".format(box))
        state.desired_agent = None
        state.parent = None

        return solution, state

    def move_blocking_boxes(self, pos, path, state):

        total_plan = []
        pos = pos[::-1]  # Reverse pos
        for i, p in enumerate(pos):
            import searchclient
            import strategy
            import heuristic

            client = searchclient.SearchClient(server_messages=None, init_state=state)

            strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
            solution, state = client.search2(strategy, pos[:i + 1])
            state.parent = None
            for sol in solution:
                total_plan.append(sol)

        return total_plan, state

    def play_plan(self, plan, wrapped=False):
        if wrapped:
            t = []
            for subp in plan:
                for p in subp:
                    t.append(p)
            plan = t

        for step in plan:
            print("\033[H\033[J")  # Stack overflow to clear screen
            print(step)  # Print state
            input()  # Wait for user input

    def find_path_with_blocking(self, goal, box, state, subgoals, agent_row=None, agent_col=
    None, config="remove"):

        if agent_row is None:
            agent_row = state.agent_row
            agent_col = state.agent_col

        path = np.zeros_like(state.walls, dtype="int")
        path_order = []

        if box:
            path[box[0]][box[1]] = 1
            path_order.append([box[0], box[1]])

        path[agent_row][agent_col] = 1
        path_order.append([agent_row, agent_col])

        import searchclient
        import strategy
        import heuristic
        from copy import deepcopy
        client = searchclient.SearchClient(server_messages=None, init_state=deepcopy(state))

        config = "remove"

        if config == "wall":
            client.initial_state.walls[client.initial_state.boxes != None] = 1

        if box:
            client.initial_state.boxes[client.initial_state.boxes != None] = None
            client.initial_state.boxes[box[0]][box[1]] = box[2].upper()
            client.initial_state.box_list = np.array([box], dtype="object")

        # Place the agent if there is an agent_input, else agent_loc is defined from state
        client.initial_state.agent_row = agent_row
        client.initial_state.agent_col = agent_col

        strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
        solution, state = client.search2(strategy, [[0, goal[:2]]])

        for sol in solution:
            if box:
                box_row = sol.box_list[0][0]
                box_col = sol.box_list[0][1]
                path[box_row][box_col] = 1
                if [box_row, box_col] not in path_order:
                    path_order.append([box_row, box_col])

            path[sol.agent_row][sol.agent_col] = 1
            if [sol.agent_row, sol.agent_col] not in path_order:
                path_order.append([sol.agent_row, sol.agent_col])

        # self.print_nmap(path, client.initial_state.walls)

        return solution, state, path, path_order

    def find_blocking_path(self, path_order, box_list, ignore_box=None):
        blocking_boxes = []
        for row, col in path_order[1:]:
            for i, box in enumerate(box_list):
                if box != ignore_box:
                    if box[0] == row and box[1] == col and row != ignore_box[0]:
                        blocking_boxes.append(i)

        return blocking_boxes

    def find_pos_blocks(self, block_box, blocking_boxes, path, path_order, state):
        import searchclient
        import strategy
        import heuristic

        client = searchclient.SearchClient(server_messages=None, init_state=state)

        # Remove earlier blocking_boxes
        for box in blocking_boxes:
            if box != block_box:
                b_row, b_col, _ = client.initial_state.box_list[box]
                client.initial_state.boxes[b_row][b_col] = None

        indicies_remove = [x for x in blocking_boxes if x != block_box]

        client.initial_state.box_list = np.delete(client.initial_state.box_list,
                                                  indicies_remove,
                                                  axis=0)

        # Get new index for the block_box
        block_box = block_box - np.sum(np.array(indicies_remove) < block_box)

        block_box_loc = list(client.initial_state.box_list[block_box])[:2]

        # Find out where to place the agent:
        for i, location in enumerate(path_order):
            if location == block_box_loc:
                break

        agent_row, agent_col = path_order[i - 1]

        client.initial_state.agent_row = agent_row
        client.initial_state.agent_col = agent_col

        strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
        solution, state = client.search2(strategy, [[block_box, path]])

        for sol in solution:
            box_row = sol.box_list[block_box][0]
            box_col = sol.box_list[block_box][1]
            path[box_row][box_col] = 1
            path[sol.agent_row][sol.agent_col] = 1

            if [box_row, box_col] not in path_order:
                path_order.append([box_row, box_col])
            if [sol.agent_row, sol.agent_col] not in path_order:
                path_order.append([sol.agent_row, sol.agent_col])

        pos = state.box_list[block_box][:2]

        return pos, path, path_order, [agent_row, agent_col]

    def topological_sort_with_cycles(self, G, labels):
        sorted_nodes = []
        while list(G.nodes):
            i, in_degree = sorted(G.in_degree, key=lambda x: x[1], reverse=False)[0]
            G.remove_node(i)
            labels.pop(i)
            sorted_nodes.append(i)
        return sorted_nodes, labels

    def print_nmap(self, nmap, walls=None):
        if walls is not None:
            walls = np.array(walls.copy(), dtype="int")
            walls[nmap != 0] = 2
            nmap = walls
        for row in nmap:
            for col in row:
                if col == 1:
                    print("1", end="")
                elif col == 2:
                    print("X", end="")
                else:
                    print(" ", end="")
            print("")

    def draw_graph(self, G, labels):
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=800, width=3)
        plt.show()

    def construct_graph(self):
        """
        Constructs the graph with no edges
        :return: Graph and labels (for plotting)
        """
        n_goals = len(self.state.goal_list)

        labels = {}
        G = nx.DiGraph()
        for i in range(n_goals):
            G.add_node(i)
            labels[i] = str(i) + ' ' + self.state.goal_list[i][2]

        return [G, labels]

    def get_adjacent_box_loc(self, box_loc):
        """
        Returns search for going from agent pos to be next to the target box
        Args:
            agent_loc: [row, col] of agent
            box_loc: [row, col] of box

        Returns: new goal_pos which is the 4 postions beside the box

        """
        routed_solution = []
        box_row, box_col = box_loc
        goal_pos = [[box_row - 1, box_col],
                    [box_row + 1, box_col],
                    [box_row, box_col + 1],
                    [box_row, box_col - 1],
                    ]

        return goal_pos

    def subgoal_routing(self, subgoals, boxes):
        """
        Routes the agent between the different subgoals
        :param subgoals: List of subgoals (goal-box) pars [(g1,b1)..(gn,bn)]
        :param boxes: List of boxes [[b1x,b1y,b1type],...]
        :return: Routed solution [[route0],[subgoal0],route[1],...]
        """
        routed_solution = []
        for goal in subgoals[0]:
            box_row = boxes[goal[0]][0]
            box_col = boxes[goal[0]][1]
            goal_pos = [[box_row - 1, box_col],
                        [box_row + 1, box_col],
                        [box_row, box_col + 1],
                        [box_row, box_col - 1]]
            routed_solution.append(goal_pos)
            routed_solution.append(goal)

        return routed_solution

    def find_best_box(self, goal, boxes, taken):
        """
        Finds the best box for a given goal
        :param goal: Target goal
        :param boxes: List of boxes [[b1x,b1y,b1type],...]
        :param taken: List of index of taken boxes [taken0, taken1,...]
        :return: Index of best box
        """
        goal_row = goal[0]
        goal_col = goal[1]
        goal_type = goal[2]

        best_dist = 1000
        best_box = None

        for i, box in enumerate(boxes):
            if box[2].lower() == goal_type and i not in taken:
                dist = self.manhatten_dist(box[0], box[1], goal_row, goal_col)
                if dist < best_dist:
                    best_box = i
                    best_dist = dist

        return best_box

    def manhatten_dist(self, row0, col0, row1, col1):
        """Find the manhatten distance between two points"""
        return np.abs(row0 - row1) + np.abs(col0 - col1)

    def solve_subgoals(self):
        """
        Use the search-client to find solution to individual subgoals,
        while not breaking already completed subgoals
        :return:Total plan of solutions to individual subgoals.
        """
        # Search for solution to the subgoals

        # TODO: RANK SUBGOAL ORDER
        total_plan = []
        state = self.state
        for i, subgoal in enumerate(self.subgoals[0]):
            import searchclient
            import strategy
            import heuristic
            client = searchclient.SearchClient(server_messages=None, init_state=state)
            if len(subgoal) > 2:
                client.initial_state.desired_agent = subgoal

            strategy = strategy.StrategyBestFirst(heuristic.Greedy(client.initial_state))
            solution, state = client.search2(strategy, self.subgoals[0][:i + 1])
            if len(subgoal) > 2:
                state.desired_agent = None

            state.parent = None
            total_plan.append(solution)

        return total_plan

    def return_solution(self):
        """
        Flattens the total_plan into a single list of states
        :return: The single list of states the constitutes the final plan
        """
        tot_sol = []
        for subsol in self.solution:
            for state in subsol:
                tot_sol.append(state)
        return tot_sol