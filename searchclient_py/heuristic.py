'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


from abc import ABCMeta, abstractmethod
from state import State
import numpy as np
import sys

class Heuristic(metaclass=ABCMeta):
    def __init__(self, initial_state: 'State'):
        pass
    
    def h(self, state: 'State') -> 'int':
        goals = np.array(state.goals)
        goals_loc = np.argwhere(goals)
        boxes = np.array(state.boxes)
        boxes_loc = np.argwhere(boxes)

        tot_dist = 0
        for box in boxes_loc:
            for goal in goals_loc:
                if boxes[box[0]][box[1]].lower() == goals[goal[0], goal[1]]:
                    tot_dist += np.abs(box[0] - goal[0]) + np.abs(box[1] - goal[1])
                    
        return tot_dist

    @abstractmethod
    def f(self, state: 'State') -> 'int': pass
    
    @abstractmethod
    def __repr__(self): raise NotImplementedError


class AStar(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State') -> 'int':
        return state.g + self.h(state)
    
    def __repr__(self):
        return 'A* evaluation'


class WAStar(Heuristic):
    def __init__(self, initial_state: 'State', w: 'int'):
        super().__init__(initial_state)
        self.w = w
    
    def f(self, state: 'State') -> 'int':
        return state.g + self.w * self.h(state)
    
    def __repr__(self):
        return 'WA* ({}) evaluation'.format(self.w)


class Greedy(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State') -> 'int':
        return self.h(state)
    
    def __repr__(self):
        return 'Greedy evaluation'

