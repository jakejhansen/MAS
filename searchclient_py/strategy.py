"""
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
"""

import heapq
from abc import ABCMeta, abstractmethod
from collections import deque
from time import perf_counter

import memory
from state import State
from heuristic import Heuristic


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
    def add_to_frontier(self, state: 'State'): raise NotImplementedError

    @abstractmethod
    def get_and_remove_leaf(self) -> 'State': raise NotImplementedError

    @abstractmethod
    def in_frontier(self, state: 'State') -> 'bool': raise NotImplementedError

    @abstractmethod
    def frontier_count(self) -> 'int': raise NotImplementedError

    @abstractmethod
    def frontier_empty(self) -> 'bool': raise NotImplementedError

    @abstractmethod
    def __repr__(self): raise NotImplementedError


class StrategyBFS(Strategy):
    """
    add_to_frontier - adds to right
    get_and_remove_leaf - pops from left

    i.e. acts as a (FIFO) queue
    """
    def __init__(self):
        super().__init__()
        self.frontier = deque()
        self.frontier_set = set()

    def add_to_frontier(self, state: 'State'):
        self.frontier.append(state)
        self.frontier_set.add(state)

    def get_and_remove_leaf(self) -> 'State':
        leaf = self.frontier.popleft()
        self.frontier_set.remove(leaf)
        return leaf

    def in_frontier(self, state: 'State') -> 'bool':
        return state in self.frontier_set

    def frontier_count(self) -> 'int':
        return len(self.frontier)

    def frontier_empty(self) -> 'bool':
        return len(self.frontier) == 0

    def __repr__(self):
        return 'Breadth-first Search'


class StrategyDFS(Strategy):
    """
    add_to_frontier - adds to right
    get_and_remove_leaf - pops from right

    i.e. acts as a (LIFO) queue or stack
    """
    def __init__(self):
        super().__init__()
        self.frontier = deque()
        self.frontier_set = set()

    def add_to_frontier(self, state: 'State'):
        self.frontier.append(state)
        self.frontier_set.add(state)

    def get_and_remove_leaf(self) -> 'State':
        leaf = self.frontier.pop()
        self.frontier_set.remove(leaf)
        return leaf

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

    def add_to_frontier(self, state: 'State', goalstate=None):
        if goalstate is None:
            heapq.heappush(self.frontier, (self.heuristic.f(state), state))
        else:
            heapq.heappush(self.frontier, (self.heuristic.f(state, goalstate), state))
        self.frontier_set.add(state)

    def get_and_remove_leaf(self) -> 'State':
        leaf = heapq.heappop(self.frontier)
        leaf = leaf[1]
        self.frontier_set.remove(leaf)
        return leaf

    def in_frontier(self, state: 'State') -> 'bool':
        return state in self.frontier_set

    def frontier_count(self) -> 'int':
        return len(self.frontier)

    def frontier_empty(self) -> 'bool':
        return len(self.frontier) == 0

    def __repr__(self):
        return 'Best-first Search (PriorityQueue) using {}'.format(self.heuristic)


