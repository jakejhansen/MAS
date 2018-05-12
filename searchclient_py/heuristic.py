'''
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
'''


from abc import ABCMeta, abstractmethod
from state import State
from pathfinder import pathfinder
import numpy as np
import action
import sys
from scipy.spatial.distance import cityblock

class Heuristic(metaclass=ABCMeta):
    def __init__(self, initial_state: 'State'):
        pass

    def manhatten_dist(self, row0, col0, row1, col1):
        """Find the manhatten distance between two points"""
        return np.abs(row0 - row1) + np.abs(col0 - col1)
    
    def h(self, state: 'State') -> 'int':
        """
        Length between all boxes and all goals.
        :param state:
        :return:
        """
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

        return tot_dist

    def h3(self, state, goalstate):
        """
        Distance between target boxes and their targetet goals in commbination with the agents
        distance to target boxes. Initially all subgoals up to subgoal n-1 is solved,
        so the distance from box to goal = 0 and thus the agents distance to this box is not
        accounted for
        :param state: Current State
        :param goalstate: list of subgoals [(box, goal), (box, goal)]
        :return: Heuristic value, a distance approximation from current state to goal state
        """

        #TODO: MAKE BETTER HEURISTIC.. Test on SAthomasappartment
        tot_dist = 0

        target_boxes = []
        for g in goalstate:
            target_boxes.append([g[0]])

        for i, subgoal in enumerate(goalstate):
            target_box = state.box_list[subgoal[0]]
            target_goal = state.goal_list[subgoal[1]]
            dist = self.manhatten_dist(target_box[0], target_box[1], target_goal[0],
                                        target_goal[1])

            tot_dist += (1*dist)

            act = state.action.action_type
            if(act != action.ActionType.Move):
                moved_box = np.argwhere(state.parent.box_list != state.box_list)[0][0]
                if moved_box not in target_boxes:
                    tot_dist += 1

            #If goal is not fulfilled, add the distance from agent to unresolved box.
            if dist > 0:
                dist_agent_box = self.manhatten_dist(target_box[0], target_box[1], state.agent_row,
                                                state.agent_col)

                tot_dist += dist_agent_box

        return tot_dist

    @abstractmethod
    def f(self, state: 'State') -> 'int': pass
    
    @abstractmethod
    def __repr__(self): raise NotImplementedError


class AStar(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State', goalstate = None) -> 'int':
        if goalstate is None:
            return self.h(state)
        else:
            return state.g + self.h3(state, goalstate)

    def __repr__(self):
        return 'A* evaluation'


class WAStar(Heuristic):
    def __init__(self, initial_state: 'State', w: 'int'):
        super().__init__(initial_state)
        self.w = w
    
    def f(self, state: 'State', goalstate = None) -> 'int':
        return state.g + self.w * self.h(state)
    
    def __repr__(self):
        return 'WA* ({}) evaluation'.format(self.w)


class Greedy(Heuristic):
    def __init__(self, initial_state: 'State'):
        super().__init__(initial_state)
    
    def f(self, state: 'State', goalstate = None) -> 'int':
        if goalstate is None:
            return self.h(state)
        else:
            return self.h3(state, goalstate)
    
    def __repr__(self):
        return 'Greedy evaluation'

