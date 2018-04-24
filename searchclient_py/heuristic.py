'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


from abc import ABCMeta, abstractmethod
from state import State
from pathfinder import pathfinder
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

    def h2(self, state: 'State') -> 'int':

        w = np.array(state.walls) != False
        b = np.array(state.boxes) != None

        condensed = np.array(w | b, dtype=int)



        goals = np.array(state.goals)
        goals_loc = np.argwhere(goals)
        boxes = np.array(state.boxes)
        boxes_loc = np.argwhere(boxes)


        #Find reachability of goals to boxes:
        reach = 0
        for goal in goals_loc:
            for box in boxes_loc:
                if boxes[box[0]][box[1]].lower() == goals[goal[0], goal[1]]:
                    condensed2 = np.copy(condensed)
                    condensed2[box[0], box[1]] = 0
                    condensed2[goal[0], goal[1]] = 0
                    path = pathfinder(condensed2, (box[0], box[1]), (goal[0], goal[1]))


                    if path:
                        reach += len(path)
                    else:
                        reach += 100

        #Find distance from goals to boxes:
        goals_to_boxes = self.h(state)

        #Add them all together:
        tot_dist = reach
        #print(tot_dist)
        #import IPython
        #IPython.embed()



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

