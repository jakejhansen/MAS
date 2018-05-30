"""
    Author: Mathias Kaas-Olsen
    Date:   2016-02-11
"""

import random
import numpy as np
from action import ALL_ACTIONS, ActionType
import networkx as nx

class Info:
    """Contains all the static level info: dimensions, colors, walls and goals."""
    def __init__(self, dims, colors=None, agent=None):
        self.dims = dims
        self.MAX_ROW, self.MAX_COL = dims

        self.colors = colors
        self.walls = np.array([[False for _ in range(self.MAX_COL)] for _ in range(self.MAX_ROW)])
        self.goals = np.array([[None for _ in range(self.MAX_COL)] for _ in range(self.MAX_ROW)])

        self.agent = agent

        self.walls_dict = {}
        self.walls_graph = None
        self.all_pairs_shortest_path_dict = {}  # apsp = all pairs shortest path

        self.walls_to_dict()
        self.dict_to_graph()
        self.graph_to_all_pairs_shortest_path_lengths_dict()

    def walls_to_dict(self):
        """Convert boolean 2D ndarray to dict of nodes."""
        nrows = self.dims[0]
        ncols = self.dims[1]
        walls = self.walls

        graph_dict = {}

        for i, row in enumerate(walls):
            if i == 0 or i == nrows - 1:  # row edge assumed to be wall -> skip
                continue
            for j, wall in enumerate(row):
                if j == 0 or j == ncols - 1:  # column edge assumed to be wall -> skip
                    continue
                if wall:
                    continue
                else:
                    node = "({},{})".format(i, j)  # node label is e.g. (4,2)

                    graph_dict[node] = self.wall_neighbors(walls, i, j)

        self.walls_dict = graph_dict

    def dict_to_graph(self):
        self.walls_graph = nx.Graph(self.walls_dict)

    def graph_to_all_pairs_shortest_path_lengths_dict(self):
        self.all_pairs_shortest_path_dict = dict(nx.all_pairs_shortest_path_length(self.walls_graph))

    def wall_neighbors(self, walls, i, j):
        neighbor_list = []
        # Neighbor relative coordinates
        n = (i - 1, j)
        s = (i + 1, j)
        e = (i, j + 1)
        w = (i, j - 1)
        # Neighbors are either 0 (non-wall) or 1 (walls)
        neighbor_n = walls[n]
        neighbor_s = walls[s]
        neighbor_w = walls[w]
        neighbor_e = walls[e]
        # If neighbor is not wall, add it as node to current
        if not neighbor_n:
            neighbor_list.append(str(n).replace(" ", ""))  # node label is e.g. (4,2)
        if not neighbor_s:
            neighbor_list.append(str(s).replace(" ", ""))
        if not neighbor_e:
            neighbor_list.append(str(e).replace(" ", ""))
        if not neighbor_w:
            neighbor_list.append(str(w).replace(" ", ""))

        return neighbor_list



class State:
    _RANDOM = random.Random(1)

    def __init__(self, copy: 'State' = None, dims=[50, 50], info=None):
        '''
        If copy is None: Creates an empty State.
        If copy is not None: Creates a copy of the copy state.
        The lists walls, boxes, and goals are indexed from top-left of the level, row-major order (row, col).
               Col 1  Col 2  Col 3  Col 4
        Row 0: (0,0)  (0,1)  (0,2)  (0,3)  ...
        Row 1: (1,0)  (1,1)  (1,2)  (1,3)  ...
        Row 2: (2,0)  (2,1)  (2,2)  (2,3)  ...
        ...
        
        For example, self.walls is a list of size [MAX_ROW][MAX_COL] and
        self.walls[2][7] is True if there is a wall at row 3, column 8 in this state.
        
        Note: The state should be considered immutable after it has been hashed, e.g. added to a dictionary!
        '''
        self.dims = dims
        State.MAX_ROW, State.MAX_COL = self.dims
        self.colors = info.colors
        self.walls = info.walls
        self.goals = info.goals
        self.desired_agent = info.agent
        self.info = info
        self._hash = None
        self.goal_list = []
        self.box_list = []
        if copy is None:
            self.agent_row = None
            self.agent_col = None

            self.boxes = np.array(
                [[None for _ in range(State.MAX_COL)] for _ in range(State.MAX_ROW)])

            self.parent = None
            self.action = None

            self.g = 0
        else:
            self.agent_row = copy.agent_row
            self.agent_col = copy.agent_col

            self.boxes = np.copy(copy.boxes)

            self.parent = copy.parent
            self.action = copy.action

            self.box_list = copy.box_list.copy()
            self.goal_list = copy.goal_list

            self.g = copy.g

            self.desired_agent = copy.desired_agent

    def make_list_representation(self):
        loc = np.argwhere(self.goals != None)
        for l in loc:
            self.goal_list.append([l[0], l[1], self.goals[l[0]][l[1]]])

        loc = np.argwhere(self.boxes != None)
        for l in loc:
            self.box_list.append([l[0], l[1], self.boxes[l[0]][l[1]].lower()])

        self.box_list = np.array(self.box_list, dtype="object")

    def get_index_from_list(self, obj, row, col):
        for i, v in enumerate(obj):
            if v[0] == row and v[1] == col:
                return i

    def get_children(self) -> '[State, ...]':
        '''
        Returns a list of child states attained from applying every applicable action in the current state.
        The order of the actions is random.
        '''
        children = []
        for action in ALL_ACTIONS:
            # Determine if action is applicable.
            new_agent_row = self.agent_row + action.agent_dir.d_row
            new_agent_col = self.agent_col + action.agent_dir.d_col

            if action.action_type is ActionType.Move:
                if self.is_free(new_agent_row, new_agent_col):
                    child = State(self, self.dims, self.info)
                    child.agent_row = new_agent_row
                    child.agent_col = new_agent_col
                    child.parent = self
                    child.action = action
                    child.g += 1
                    children.append(child)
            elif action.action_type is ActionType.Push:
                if self.box_at(new_agent_row, new_agent_col):
                    new_box_row = new_agent_row + action.box_dir.d_row
                    new_box_col = new_agent_col + action.box_dir.d_col
                    if self.is_free(new_box_row, new_box_col):
                        child = State(self, self.dims, self.info)
                        child.agent_row = new_agent_row
                        child.agent_col = new_agent_col
                        child.boxes[new_box_row][new_box_col] = self.boxes[new_agent_row][
                            new_agent_col]
                        child.boxes[new_agent_row][new_agent_col] = None
                        child.parent = self
                        child.action = action
                        child.g += 1

                        idx = self.get_index_from_list(child.box_list, new_agent_row, new_agent_col)
                        child.box_list[idx][0] = new_box_row
                        child.box_list[idx][1] = new_box_col
                        children.append(child)

            elif action.action_type is ActionType.Pull:
                if self.is_free(new_agent_row, new_agent_col):
                    box_row = self.agent_row + action.box_dir.d_row
                    box_col = self.agent_col + action.box_dir.d_col
                    if self.box_at(box_row, box_col):
                        child = State(self, self.dims, self.info)
                        child.agent_row = new_agent_row
                        child.agent_col = new_agent_col
                        child.boxes[self.agent_row][self.agent_col] = self.boxes[box_row][box_col]
                        child.boxes[box_row][box_col] = None
                        child.parent = self
                        child.action = action
                        child.g += 1

                        idx = self.get_index_from_list(child.box_list, box_row, box_col)
                        child.box_list[idx][0] = self.agent_row
                        child.box_list[idx][1] = self.agent_col
                        children.append(child)

        State._RANDOM.shuffle(children)
        return children

    def is_initial_state(self) -> 'bool':
        return self.parent is None

    def is_goal_state(self) -> 'bool':

        """ #Original Code
        for row in range(State.MAX_ROW):
            for col in range(State.MAX_COL):
                goal = self.goals[row][col]
                box = self.boxes[row][col]
                if goal is not None and (box is None or goal != box.lower()):
                    return False
        return True
        """

        # Convert arrays to np and filter out fields with goals and boxes
        # ontop of goal fields.
        g_list = self.goals[self.goals != None]
        b_list = self.boxes[self.goals != None]

        # Agent is not at desired location
        if self.desired_agent != None:
            if not (self.agent_row == self.desired_agent[0] and self.agent_col ==
                    self.desired_agent[1]):
                return False

        # If any goal does not have a box, we havn't reached the goal state
        if not (np.any(np.equal(b_list, None))):
            if np.sum(g_list) == np.sum(b_list).lower():
                return True

        return False

    def is_goal_state2(self, goal_state, allow_pull = True) -> 'bool':
        """ Second iteration of finding goal by using the box_list

            Input: goal_state - List of box_index and desired location
        """

        if goal_state is not None:
            for subgoal in goal_state:
                if len(subgoal) <= 2:

                    if type(subgoal[1]) == list:
                        target_box = self.box_list[subgoal[0]]
                        target_goal = subgoal[1]
                        if target_box[0] != target_goal[0] or target_box[1] != target_goal[1]:
                            return False
                        if allow_pull != True:
                            if self.action:
                                if self.action.action_type == ActionType.Pull:
                                    return False

                    else:
                        target_box = self.box_list[subgoal[0]]
                        path = subgoal[1]
                        if self.action:
                            if self.action.action_type == ActionType.Pull:
                                return False
                        if path[target_box[0], target_box[1]] == 1:
                            return False


        if self.desired_agent is not None:
            for pos in self.desired_agent:
                if self.agent_row == pos[0] and self.agent_col == pos[1]:
                    return True

            return False

        return True

    def is_free(self, row: 'int', col: 'int') -> 'bool':
        return not self.walls[row][col] and self.boxes[row][col] is None

    def box_at(self, row: 'int', col: 'int') -> 'bool':
        return self.boxes[row][col] is not None

    def extract_plan(self) -> '[State, ...]':
        plan = []
        state = self
        while not state.is_initial_state():
            plan.append(state)
            state = state.parent
        plan.reverse()
        return plan

    def __hash__(self):
        if self._hash is None:
            # prime = 31
            # _hash = 1
            # _hash = _hash * prime + self.agent_row
            # _hash = _hash * prime + self.agent_col
            # _hash = _hash * prime + hash(self.boxes.tostring())
            # _hash = _hash * prime + hash(self.goals.tostring())
            # _hash = _hash * prime + hash(self.walls.tostring())
            # self._hash = _hash
            # Hash a tuple of all the relevant parameters, no risk of overflow.
            self._hash = hash((self.agent_row,
                               self.agent_col,
                               self.boxes.tostring(),
                               self.goals.tostring(),
                               self.walls.tostring()))
        return self._hash

    def __eq__(self, other):

        if self._hash == other._hash:
            return True
        return False

        """
        if self is other: return True
        if not isinstance(other, State): return False
        if self.agent_row != other.agent_row: return False
        if self.agent_col != other.agent_col: return False
        if np.any(self.boxes != other.boxes): return False
        if np.any(self.goals != other.goals): return False
        if np.any(self.walls != other.walls): return False
        return True
        """

    def __repr__(self):
        lines = []
        for row in range(State.MAX_ROW):
            line = []
            for col in range(State.MAX_COL):
                if self.boxes[row][col] is not None:
                    line.append(self.boxes[row][col])
                elif self.goals[row][col] is not None:
                    line.append(self.goals[row][col])
                elif self.walls[row][col] == True:
                    line.append('+')
                elif self.agent_row == row and self.agent_col == col:
                    line.append('0')
                else:
                    line.append(' ')
            lines.append(''.join(line))
        return '\n'.join(lines)

    def __lt__(self, other):
        return True
