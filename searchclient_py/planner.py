from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from corner_finder import corner_finder
from pathfinder import pathfinder


class Custom():
    def __init__(self, init_state, info):
        self.init_state = init_state
        self.state = init_state
        self.info = info

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
        # G, completed_goals, completed_goals_index = \
        #     self.compute_goalgraph_edges(self.state, G, labels, completed_goals, completed_goals_index, boxes)


        taken = []
        subgoals = []
        tot_solution = []

        for i in range(len(G)): #Run through all goals
            completed_goals, completed_goals_index = self.sample_next_goal(self.state, G, labels, completed_goals, completed_goals_index,
                                self.state.box_list.tolist())
            goal = self.state.goal_list[completed_goals_index[-1]]
            box_id = self.find_best_box(goal, boxes, taken)
            box = self.state.box_list.tolist()[box_id]

            # # Move agent to box, imaginarily.
            # subgoals.append(self.get_adjacent_box_loc([box[0], box[1]]))
            # solution, state_imaginary = self.move_agt_next_to_box(deepcopy(self.state),
            #                                                       box,
            #                                                       subgoals)


            state_imaginary = deepcopy(self.state)
            state_imaginary.parent = None
            state_imaginary.agent_row = box[0]
            state_imaginary.agent_col = box[1]
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
            box = self.state.box_list.tolist()[box_id]
            solution, temp_state = self.box_to_goal(box, box_id, goal, subgoals)

            tot_solution.append(solution)
            self.state = temp_state

            taken.append(box_id)

        return tot_solution

    def box_to_goal(self, box, box_id, goal, subgoals):
        """
        Find a way to move the box to a goal
        Args:
            box: [row, col, type]
            box_id: index of the box in the state.box_list
            goal: [row, col, type]
            subgoals: List of subgoals, used to ensure solution maintains earlier solved subgoals.

        Returns:
            solution: List of states that constitues a solution from input state
            state: final state after solution has been applied to input state.
        """
        # Move box to goal
        import searchclient
        import strategy
        import heuristic

        len_wall = 10000
        len_normal = 10000

        # Calculate length of solution with other boxes set to walls
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

        solution_wall = None  # Flag for use later
        if self.path_is_clear(client.initial_state,
                              [box[0], box[1]],
                              goal[:2],
                              box_ignore=box):
            solution_wall, temp_state_wall = client.search2(strategy, [[0, goal[:2]]],
                                                            msg="Box {} to Goal {} - No other "
                                                                "boxes".format(box,
                                                                               goal))

            len_wall = len(solution_wall)

            # Remove old pos
            init_backup.boxes[init_backup.box_list[box_id][0]][
                init_backup.box_list[box_id][1]] = None
            # Put at new pos
            init_backup.boxes[temp_state_wall.box_list[0][0]][temp_state_wall.box_list[0][1]] \
                = \
                temp_state_wall.box_list[0][
                    2].upper()

            init_backup.box_list[box_id] = temp_state_wall.box_list[0]
            init_backup.agent_col = temp_state_wall.agent_col
            init_backup.agent_row = temp_state_wall.agent_row
            temp_state_wall = init_backup

        # Do the other search
        import searchclient
        import strategy
        import heuristic
        client = searchclient.SearchClient(server_messages=None,
                                           init_state=deepcopy(self.state))
        strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
        if solution_wall == None:
            solution_normal, temp_state_normal = client.search2(strategy, subgoals,
                                                                msg="Box {} to Goal{} - With other "
                                                                    "boxes".format(box, goal))

        else:
            solution_normal, temp_state_normal = client.search2(strategy, subgoals,
                                                                msg="Box {} to Goal{} - With other "
                                                                    "boxes".format(box, goal),
                                                                max_time=5)

        if solution_normal:
            len_normal = len(solution_normal)
            temp_state_normal.parent = None

        if len_wall < (1.5 * len_normal) and solution_wall:
            solution = solution_wall
            temp_state = temp_state_wall

        else:
            solution = solution_normal
            temp_state = temp_state_normal

        return solution, temp_state


    def compute_goalgraph_edges(self, state, G, labels, completed_goals, completed_goals_index,
                                boxes):
        """
        Not used atm - Can find goal order from initial condition (not dynamic)
        """
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


    def sample_next_goal(self, state, G, labels, completed_goals, completed_goals_index,
                                boxes):
        """
        Takes the list of currently completed goals and finds the next goal to complete
        Args:
            state: type: State
            G: Graph
            labels: labels of graph
            completed_goals: With goals has been completed [[row, col, type], [row, col, type]..]
            completed_goals_index: index of completed goals [g_index0, g_index3, ...]
            boxes: list of boxes [[row, col, type],[row,col,type]]

        Returns:
            completed_goals: Which goals has been completed (one more than the input)
            completed_goals_index: Index of completed goals (one more than the input)
        """
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

                        return completed_goals, completed_goals_index

            counter += 1
            if counter > 100:
                completed_goals_index, labels = self.topological_sort_with_cycles(G, labels)
                break

        return completed_goals, completed_goals_index

    def path_is_clear(self, state, start, finish, box_ignore=None):
        """
        Finds out of the path from start to finish is clear in the current state.
        Args:
            state: type: State
            start: [start_row, start,col]
            finish: [finish_row, finish_col]
            box_ignore: [row, col, type] - Default None, box to be ignored

        Returns:
            path: If path is clear, a list of visited fields is returned, else it returns False
        """
        nmap = state.walls.astype('int')
        nmap[state.boxes != None] = 1
        if box_ignore:
            nmap[box_ignore[0]][box_ignore[1]] = 0

        v = pathfinder(nmap, (start[0], start[1]),
                       (finish[0], finish[1]))
        if v:
            return v

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

        #TODO: Rewrite so the agent can move blocking boxes

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
        """
        Figure out how to move the blocking boxes out of the way
        Args:
            pos: List of position for the blocking boxes
            path: Not used ATM
            state: Type: State

        Returns:
            total_plan: Combined plan for moving all boxes out of the path from input state
            state: Final state after plan has been carried out
        """
        total_plan = []
        pos = pos[::-1]  # Reverse pos
        for i, p in enumerate(pos):
            import searchclient
            import strategy
            import heuristic

            client = searchclient.SearchClient(server_messages=None, init_state=state)

            strategy = strategy.StrategyBestFirst(heuristic.AStar(client.initial_state))
            solution, state = client.search2(strategy, pos[:i + 1], allow_pull=False,
                                             msg="Box {} out of path".format(self.state.box_list[
                                                                                 p[0]]))
            state.parent = None
            for sol in solution:
                total_plan.append(sol)

        return total_plan, state

    def play_plan(self, plan, wrapped=False):
        """
        Plays a given plan in the console, enter to continue
        Args:
            plan: A list of states that constitutes a solution
            wrapped: If the plan is a list of lists [solution1, solution2], set to True to
                flatten list

        Returns:
            None: Prints the plan to console, enter to continue
        """

        if wrapped: #Unwrap plan if needed
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
        """
        Find the path agent to box to goal (agent can be next to box) where all other boxes are
        removed
        Args:
            goal: Target goal [row, col, type]
            box: Target box [row, col, type]
            state: Type: State
            subgoals: List og subgoals [subgoal1, subgoal2]. Can be box, location pairs and other things
            agent_row / agent_col: Default is None, but can be set outside if we want to find the
            path without actually moving the agent there with state
            config: If other boxes should be removed or other boxes should be set to walls

        Returns:
            solution: List of states in solution
            state: Final state after solution is found:
            path: 2d map with 1's where the path is
            path_order: List of visited fields in path, ordered by first visited [field0, field1,..]
        """

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
        """
        Finds the boxes which are in the path and gives them in the order that they appear seen
        from goal --> box
        Args:
            path_order: List of visited fields in path, ordered by first visited [field0, field1,..]
            box_list: List of boxes [[row, col, type]_0, [row, col, type]_1]
            ignore_box: Box to be ignored when finding blocking boxes [row, col, type]

        Returns:
            blocking_boxes: List of the blocking boxes in the path
        """
        blocking_boxes = []
        for row, col in path_order[1:]:
            for i, box in enumerate(box_list):
                if box != ignore_box:
                    if box[0] == row and box[1] == col and row != ignore_box[0]:
                        blocking_boxes.append(i)

        return blocking_boxes

    def find_pos_blocks(self, block_box, blocking_boxes, path, path_order, state):
        """
        Takes the list of blocking boxes and finds where to place them out of path
        Args:
            block_box: The box that we want to move out of pathj
            blocking_boxes: List of all blocking boxes
            path: 2d Map of the path
            path_order: List of visited fields in path, ordered by first visited [field0, field1,..]
            state: Type: State

        Returns:
            pos: Designated position of box
            path: Union(Path, Path for moving box out of old path). Essentially expands the path
            path_order: Same as input, where we add to the path_order
            [agent_row, agent_col]: End position of the agent

        """
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
        solution, state = client.search2(strategy, [[block_box, path]],
                                         msg="Blocking box {} new position".format(block_box))

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
        """
        Runs a topological sort on the graph using cycles
        Args:
            G: Graph
            labels: Labels of graph

        Returns:
            sorted_nodes: Nodes in sorted order
            labels: New labels
        """
        sorted_nodes = []
        while list(G.nodes):
            i, in_degree = sorted(G.in_degree, key=lambda x: x[1], reverse=False)[0]
            G.remove_node(i)
            labels.pop(i)
            sorted_nodes.append(i)
        return sorted_nodes, labels

    def print_nmap(self, nmap, walls=None):
        """
        Small function that can print a path in readable format. Optionally is can also plot the
        walls
        Args:
            nmap: Binary map
            walls: state.walls

        Returns:
            None: Prints the nmap to console
        """
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
        """
        Draws the graph
        Args:
            G: Graph
            labels: Labels of graph
        """
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
                # dist = self.manhattan_dist(box[0], box[1], goal_row, goal_col)
                dist = self.shortest_path_dist(box[0], box[1], goal_row, goal_col)
                if dist < best_dist:
                    best_box = i
                    best_dist = dist

        return best_box

    def manhattan_dist(self, row0, col0, row1, col1):
        """Find the manhatten distance between two points"""
        return np.abs(row0 - row1) + np.abs(col0 - col1)

    def shortest_path_dist(self, row0, col0, row1, col1):
        """Look up shortest path between two points."""
        path = self.info.all_pairs_shortest_path_dict["({},{})".format(row0, col0)]["({},{})".format(row1, col1)]
        return len(path) - 1  # subtracting one, i.e. not counting the starting position in list

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