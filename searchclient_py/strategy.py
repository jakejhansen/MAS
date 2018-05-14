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

        def print_nmap(nmap):
            for row in nmap:
                for col in row:
                    if col == 1:
                        print("1", end="")
                    else:
                        print(" ", end="")
                print("")

        completeable_boxes = defaultdict(set)
        #Run through goals in graph
        for i in G:
            goal = self.state.goal_list[i]
            #Check if completing the goal will block the completion of other goals
            nmap = self.state.walls.astype('int')
            nmap[goal[0]][goal[1]] = 1 # Pretend that current goal i is occupied -> make into wall
            # For every other goal than goal[i], see if it's achievable when goal[i] is occupied
            for j, other_goal in enumerate(self.state.goal_list):
                if j != i:
                    goal_completable = False
                    for b_n, box in enumerate(boxes):
                        if box[2] == other_goal[2]: # same type
                            v = pathfinder(nmap, (box[0], box[1]), (other_goal[0], other_goal[1]))
                            if v:
                                goal_completable = True  # goal achievable by *some* box of same type
                                completeable_boxes[j].add(b_n) # Goal of index j can be achieved
                                                                # specifically by box of index b_n
                    if goal_completable == False:
                        G.add_edge(j, i)

        # Fix completable_goals to also include corner goals that can be missed / excluded otherwise
        # (such as bottom goal in boxesOfHanoi.lvl), i.e. candidate goals have zero in-degree
        nmap = self.state.walls.astype('int')
        for in_degree, i in sorted(G.in_degree, key=lambda x: x[1], reverse=True):
            if in_degree == 0:
                goal = goals[i]
                for b_n, box in enumerate(boxes):
                    if box[2] == goal[2]:
                        v = pathfinder(nmap, (box[0], box[1]), (goal[0], goal[1]))
                        if v:
                            completeable_boxes[j].add(b_n)


        #Goal Assignment (on graph)
        taken = [] #List of taken boxes, initialy empty
        gb_pair = [] #List of goal-box pairs
        for in_degree, i in sorted(G.in_degree, key=lambda x: x[1], reverse=True):
            goal = self.state.goal_list[i]
            box = self.find_best_box(goal, boxes, taken) #TODO: MAKE IT USE THE COMPLETEABLE BOXES
            taken.append(box)  # Mark the box as taken
            gb_pair.append([box, i])

        subgoals.append(gb_pair)


        """
        #Goal Assignment
        taken = []
        gb_pair = []
        for i, goal in enumerate(goals):
            goal_row = goal[0]
            goal_col = goal[1]
            goal_type = goal[2]

            #Find best box for the goal
            box = self.find_best_box(goal, boxes, taken)
            taken.append(box) #Mark the box as taken
            gb_pair.append([box, i]) #

        subgoals.append(gb_pair)
        """

        #Route the agent to go to the target box
        subgoals[0] = self.subgoal_routing(subgoals, boxes)

        return subgoals


    def draw_graph(self, G, labels):
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos=pos, labels=labels, with_labels=True, node_size=800, width = 3)
        plt.show()

    def construct_graph(self):
        n_goals = len(self.state.goal_list)

        labels = {}
        G = nx.DiGraph()
        for i in range(n_goals):
            G.add_node(i)
            labels[i] = str(i) + ' ' + self.state.goal_list[i][2]

        return [G, labels]

    def subgoal_routing(self, subgoals, boxes):
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

    #Route the subgoals

    def find_best_box(self, goal, boxes, taken):
        """Finds the best box for a given goal"""
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
        tot_sol = []
        for subsol in self.solution:
            for state in subsol:
                tot_sol.append(state)
        return tot_sol
