'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


from abc import ABCMeta, abstractmethod
from collections import deque
from time import perf_counter

import memory
import heapq
import numpy as np
import copy

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
        return '#Explored: {:4}, #Frontier: {:3}, Time: {:3.2f} s, Alloc: {:4.2f} MB, MaxAlloc: {:4.2f} MB'.format(self.explored_count(), self.frontier_count(), self.time_spent(), memory.get_usage(), memory.max_usage)
    
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
    
    def add_to_frontier(self, state: 'State'):
        heapq.heappush(self.frontier, (self.heuristic.f(state), state))
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
        self.subgoals = self.initStrategy()

        self.solution = self.solve_subgoals()



    def initStrategy(self):
        subgoals = []
        goals = self.state.goal_list
        boxes = self.state.box_list
        b_array = np.array(boxes)


        def find_best_box(goal, boxes, taken):
            """Finds the best box for a given goal"""
            goal_row = goal[0]
            goal_col = goal[1]
            goal_type = goal[2]

            best_dist = 1000
            best_box = None

            for i, b in enumerate(boxes):
                if b[2].lower() == goal_type and i not in taken:
                    dist = manhatten_dist(b[0], b[1], goal_row, goal_col)
                    if dist < best_dist:
                        best_box = i
                        best_dist = dist

            return best_box


        def manhatten_dist(row0, col0, row1, col1):
            """Find the manhatten distance between two points"""
            return np.abs(row0 - row1) + np.abs(col0 - col1)

        #Goal Assignment
        taken = []
        gb_pair = []
        for i, goal in enumerate(goals):
            goal_row = goal[0]
            goal_col = goal[1]
            goal_type = goal[2]

            #Find best box for the goal
            b = find_best_box(goal, boxes, taken)
            taken.append(b) #Mark the box as taken

            gb_pair.append([b, i]) #

        subgoals.append(gb_pair)

        return subgoals

    def solve_subgoals(self):
        #Search for solution to the subgoals

        total_plan = []
        state = self.state
        for i, subgoal in enumerate(self.subgoals[0]):
            if i > 0:
                break
            import searchclient
            import strategy
            import heuristic
            client = searchclient.SearchClient(server_messages = None, init_state =
            copy.deepcopy(state))
            strategy = strategy.StrategyBestFirst(heuristic.Greedy(client.initial_state))
            solution, state = client.search2(strategy, self.subgoals[0])
            state.parent = None
            #state.action = None
            total_plan.append(solution)

        return total_plan

    def return_solution(self):
        tot_sol = []
        for subsol in self.solution:
            for state in subsol:
                tot_sol.append(state)
        return tot_sol
