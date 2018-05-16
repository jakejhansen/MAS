'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


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
import pickle

class Strategy(metaclass=ABCMeta):
    def __init__(self):
        self.explored = set()
        self.start_time = perf_counter()
    
    def add_to_explored(self, state: 'State'):
        self.explored.add(state)
    
    def is_explored(self, state: 'State') -> 'bool':
        return state in self.explored
    
    def explored_count(self) -> 'int':
        return len(self.explored)
    
    def time_spent(self) -> 'float':
        return perf_counter() - self.start_time
    
    def search_status(self) -> 'str':
        return '#Explored: {:4},' \
               '#Frontier: {:3},' \
               'Time: {:3.2f} s,' \
               'Alloc: {:4.2f} MB,' \
               'MaxAlloc: {:4.2f} MB'.format(self.explored_count(),
                                             self.frontier_count(),
                                             self.time_spent(),
                                             memory.get_usage(),
                                             memory.max_usage)
    
    @abstractmethod
    def get_and_remove_leaf(self) -> 'State': raise NotImplementedError
    
    @abstractmethod
    def add_to_frontier(self, state: 'State'): raise NotImplementedError
    
    @abstractmethod
    def in_frontier(self, state: 'State') -> 'bool': raise NotImplementedError
    
    @abstractmethod
    def frontier_count(self) -> 'int': raise NotImplementedError
    
    @abstractmethod
    def frontier_empty(self) -> 'bool': raise NotImplementedError
    
    @abstractmethod
    def __repr__(self): raise NotImplementedError


class StrategyBFS(Strategy):
    def __init__(self):
        super().__init__()
        self.frontier = deque()
        self.frontier_set = set()
    
    def get_and_remove_leaf(self) -> 'State':
        leaf = self.frontier.popleft()
        self.frontier_set.remove(leaf)
        return leaf
    
    def add_to_frontier(self, state: 'State'):
        self.frontier.append(state)
        self.frontier_set.add(state)
    
    def in_frontier(self, state: 'State') -> 'bool':
        return state in self.frontier_set
    
    def frontier_count(self) -> 'int':
        return len(self.frontier)
    
    def frontier_empty(self) -> 'bool':
        return len(self.frontier) == 0
    
    def __repr__(self):
        return 'Breadth-first Search'


class StrategyDFS(Strategy):
    def __init__(self):
        super().__init__()
        self.frontier = deque()
        self.frontier_set = set()
    
    def get_and_remove_leaf(self) -> 'State':
        leaf = self.frontier.pop()
        self.frontier_set.remove(leaf)
        return leaf
    
    def add_to_frontier(self, state: 'State'):
        self.frontier.append(state)
        self.frontier_set.add(state)
    
    def in_frontier(self, state: 'State') -> 'bool':
        return state in self.frontier_set
    
    def frontier_count(self) -> 'int':
        return len(self.frontier)
    
    def frontier_empty(self) -> 'bool':
        return len(self.frontier) == 0
    
    def __repr__(self):
        return 'Depth-first Search'


class StrategyBestFirst(Strategy):
    def __init__(self, heuristic: 'Heuristic'):
        super().__init__()
        self.heuristic = heuristic
        self.frontier_set = set()
        self.frontier = []
    
    def get_and_remove_leaf(self) -> 'State':
        leaf = heapq.heappop(self.frontier)
        leaf = leaf[1]
        self.frontier_set.remove(leaf)
        return leaf
    
    def add_to_frontier(self, state: 'State', goalstate = None):
        if goalstate is None:
            heapq.heappush(self.frontier, (self.heuristic.f(state), state))
        else:
            heapq.heappush(self.frontier, (self.heuristic.f(state, goalstate), state))
        self.frontier_set.add(state)
    
    def in_frontier(self, state: 'State') -> 'bool':
        return state in self.frontier_set
    
    def frontier_count(self) -> 'int':
        return len(self.frontier)
    
    def frontier_empty(self) -> 'bool':
        return len(self.frontier) == 0
    
    def __repr__(self):
        return 'Best-first Search (PriorityQueue) using {}'.format(self.heuristic)


class Custom():
    def __init__(self, init_state):
        self.init_state = init_state
        self.state = init_state
        self.corner_list = corner_finder(self.state.walls.astype('int'), self.state.goals)[0]

        self.subgoals = self.initStrategy()

        self.solution = self.solve_subgoals()



    def initStrategy(self):
        subgoals = []
        goals = self.state.goal_list
        boxes = self.state.box_list
        boxes = boxes.tolist()

        G, labels = self.construct_graph()


        completeable_boxes = defaultdict(set)
        completed_goals = []
        completed_goals_index = []

        #Run through goals in graph
        while len(completed_goals) != len(G.nodes):
            for i in G:
                if i not in completed_goals_index:
                    goal = self.state.goal_list[i]
                    #Check if completing the goal will block the completion of other goals
                    nmap = self.state.walls.astype('int')
                    nmap[goal[0]][goal[1]] = 1 # Pretend that current goal i is occupied -> make into wall
                    for g in completed_goals:
                        nmap[g[0]][g[1]] = 1
                    all_goals_completeable = True
                    for j, other_goal in enumerate(self.state.goal_list):
                        if j != i and j not in completed_goals_index:
                            goal_completable = False
                            for b_n, box in enumerate(boxes):
                                if box[2] == other_goal[2]: # same type
                                    v = pathfinder(nmap, (box[0], box[1]),
                                                   (other_goal[0], other_goal[1]))
                                    if v:
                                        goal_completable = True  # goal achievable by *some* box of same type
                                        completeable_boxes[j].add(b_n) # Goal of index j can be achieved
                                                                        # specifically by box of index b_n
                            if goal_completable == False:
                                G.add_edge(j, i)
                                all_goals_completeable = False


                    if all_goals_completeable:
                        completed_goals.append(goal)
                        completed_goals_index.append(i)


        sorted_nodes, labels = self.topological_sort_with_cycles(G, labels)

        #Goal Assignment (on graph)
        taken = [] #List of taken boxes, initialy empty
        gb_pair = [] #List of goal-box pairs
        for i in completed_goals_index:
            goal = self.state.goal_list[i]
            box = self.find_best_box(goal, boxes, taken) #TODO: MAKE IT USE THE COMPLETEABLE BOXES

            from copy import deepcopy
            solution, state, path, path_order = self.find_path_with_blocking(goal,
                                                                 deepcopy(self.state.box_list[box]),
                                                                 deepcopy(self.state),
                                                                 subgoals,
                                                                 agent_row=23,
                                                                 agent_col=6)

            #Find if there is any boxes blocking the path
            blocking_boxes = self.find_blocking_path(path_order, boxes, ignore_box = boxes[box])

            if len(blocking_boxes) > 0:
                pos_blocking_boxes = []
                for block_box in blocking_boxes[::-1]:
                    pos, path = self.find_pos_blocks(block_box,
                                                               blocking_boxes,
                                                               path,
                                                               deepcopy(self.init_state),
                                                               agent_row = boxes[block_box][0]-1,
                                                               agent_col = boxes[block_box][1])

                    pos_blocking_boxes.append([block_box, pos.tolist()])

                #Move the blocking boxes to their respective positions
                plan_move_box = self.move_blocking_boxes(pos_blocking_boxes,
                                                         path,
                                                         deepcopy(self.init_state))

            taken.append(box)  # Mark the box as taken
            gb_pair.append([box, i])

        subgoals.append(gb_pair)




        #Route the agent to go to the target box
        subgoals[0] = self.subgoal_routing(subgoals, boxes)

        return subgoals

    def move_blocking_boxes(self, pos, path, state):

        total_plan = []
        pos = pos[::-1] #Reverse pos
        for i, p in enumerate(pos):
            import searchclient
            import strategy
            import heuristic

            client = searchclient.SearchClient(server_messages=None, init_state=state)
            if i == 0:
                client.initial_state.agent_row = 14
                client.initial_state.agent_col = 15

            strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
            solution, state = client.search2(strategy, pos[:i+1])
            state.parent = None
            for sol in solution:
                total_plan.append(sol)

        return total_plan

    def play_plan(self, plan):
        for step in plan:
            print("\033[H\033[J")  # Stack overflow to clear screen
            print(step) #Print state
            input() #Wait for user input

    def find_path_with_blocking(self, goal, box, state, subgoals, agent_row, agent_col):
        path = np.zeros_like(state.walls, dtype="int")
        path_order = []

        path[box[0]][box[1]] = 1
        path[agent_row][agent_col] = 1
        path_order.append([box[0], box[1]])
        path_order.append([agent_row, agent_col])

        import searchclient
        import strategy
        import heuristic
        client = searchclient.SearchClient(server_messages=None, init_state=state)


        config = "remove"

        if config == "wall":
            client.initial_state.walls[client.initial_state.boxes != None] = 1

        client.initial_state.boxes[client.initial_state.boxes != None] = None
        client.initial_state.boxes[box[0]][box[1]] = box[2].upper()
        client.initial_state.box_list = np.array([box], dtype="object")

        client.initial_state.agent_row = agent_row
        client.initial_state.agent_col = agent_col




        strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
        #solution, state = client.search2(strategy, subgoals[0][:1])
        solution, state = client.search2(strategy, [[0, 5]])

        for sol in solution:
            box_row = sol.box_list[0][0]
            box_col = sol.box_list[0][1]
            path[box_row][box_col] = 1
            path[sol.agent_row][sol.agent_col] = 1

            if [box_row, box_col] not in path_order:
                path_order.append([box_row, box_col])
            if [agent_row, agent_col] not in path_order:
                path_order.append([agent_row, agent_col])

        self.print_nmap(path, client.initial_state.walls)

        return solution, state, path, path_order

    def find_blocking_path(self, path_order, box_list, ignore_box = None):
        blocking_boxes = []
        for row, col in path_order[1:]:
            for i, box in enumerate(box_list):
                if box[0] == row and box[1] == col and row != ignore_box[0] and col != \
                        ignore_box[1]:
                    blocking_boxes.append(i)

        return blocking_boxes

    def find_pos_blocks(self, block_box, blocking_boxes, path, state, agent_row, agent_col):
        import searchclient
        import strategy
        import heuristic

        client = searchclient.SearchClient(server_messages=None, init_state=state)

        for box in blocking_boxes:
            if box != block_box:
                b_row, b_col, _ = client.initial_state.box_list[box]
                client.initial_state.boxes[b_row][b_col] = None

        indicies_remove = [x for x in blocking_boxes if x != block_box]

        client.initial_state.box_list = np.delete(client.initial_state.box_list,
                                                  indicies_remove,
                                                  axis = 0)

        #Get new index for the block_box
        block_box = block_box - np.sum(np.array(indicies_remove) < block_box)

        client.initial_state.agent_row = agent_row
        client.initial_state.agent_col = agent_col

        strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
        solution, state = client.search2(strategy, [[block_box, path]])

        for sol in solution:
            box_row, box_col, _ = sol.box_list[block_box]
            path[box_row][box_col] = 1
            path[sol.agent_row][sol.agent_col] = 1

        pos = state.box_list[block_box][:2]

        return pos, path


    def topological_sort_with_cycles(self, G, labels):
        sorted_nodes = []
        while list(G.nodes):
            i, in_degree = sorted(G.in_degree, key=lambda x: x[1], reverse=False)[0]
            G.remove_node(i)
            labels.pop(i)
            sorted_nodes.append(i)
        return sorted_nodes, labels

    def print_nmap(self, nmap, walls = None):
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
        nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=800, width = 3)
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
                        [box_row +1, box_col],
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
        Use the search-client to find solution to individual subgoals, while not breaking already
        completed subgoals
        :return:Total plan of solutions to individual subgoals.
        """
        #Search for solution to the subgoals

        #TODO: RANK SUBGOAL ORDER
        total_plan = []
        state = self.state
        for i, subgoal in enumerate(self.subgoals[0]):
            import searchclient
            import strategy
            import heuristic
            client = searchclient.SearchClient(server_messages = None, init_state = state)
            if len(subgoal) > 2:
                client.initial_state.desired_agent = subgoal

            strategy = strategy.StrategyBestFirst(heuristic.Greedy(client.initial_state))
            solution, state = client.search2(strategy, self.subgoals[0][:i+1])
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
