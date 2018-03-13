package searchclient;

import java.util.ArrayList;
import java.util.List;
import java.util.Comparator;

public abstract class Heuristic implements Comparator<Node> {
	private List<int[]> goals;
	
	public Heuristic(Node initialState) {
		
		//Get the location of all goals
		goals = new ArrayList<>();
		for (int i = 0; i < initialState.MAX_ROW; i++) {
			for (int j = 0; j < initialState.MAX_COL; j++) {
				char goal = initialState.goals[i][j];
				if ('a' <= goal && goal <= 'z') {
					goals.add(new int[] {i,j});
				}
			}
			
		}
		//System.err.println(goals.size());
	}

	public int h(Node n) {
		int rows = n.MAX_ROW;
		int cols = n.MAX_COL;
		
		/*
		//Loop through boxes
		int totdist = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				char box = n.boxes[i][j];
				if ('A' <= box && box <= 'Z') { //box
					int minDist = rows * cols;
					for (int ii = 0; ii < goals.size(); ii++) {
						int rowGoal = goals.get(ii)[0];
						int colGoal = goals.get(ii)[1];
						if(n.goals[rowGoal][colGoal] == Character.toLowerCase(box)) {
							//We found a valid goal of the box
							int dist = Math.abs(i - rowGoal) + Math.abs(j - colGoal);
							if (dist < minDist) {
								minDist = dist;
							}
						}
					}
					totdist += minDist;
				}
			}
		}
		return totdist;
		*/
		
		/*
		// Distance from goal closest to agent to box and box -> agent
		
		//Find closest goal to agent
		int minDistGoal = rows * cols;
		int rowGoalBest = 0;
		int colGoalBest = 0;
		for (int i = 0; i < goals.size(); i++) {
			int rowGoal = goals.get(i)[0];
			int colGoal = goals.get(i)[1];
			int distGoal = Math.abs(rowGoal - n.agentRow) + Math.abs(colGoal - n.agentCol);
			// If distance is the smallest we have seen and there is not already a correct box on the goal
			if (distGoal < minDistGoal && Character.toLowerCase(n.boxes[rowGoal][colGoal]) != n.goals[rowGoal][colGoal]) {
				minDistGoal = distGoal;
				rowGoalBest = rowGoal;
				colGoalBest = colGoal;
			}
		}
		//Find closest box to goal
		int minDistBox = rows * cols;
		int rowBestBox = 0;
		int colBestBox = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				char box = n.boxes[i][j];
				if (Character.toLowerCase(box) == n.goals[rowGoalBest][colGoalBest]) {
					int distBox = Math.abs(i - rowGoalBest) + Math.abs(j - colGoalBest);
					if (distBox < minDistBox){
						minDistBox = distBox;
						rowBestBox = i;
						colBestBox = j;
					}
				}
			}
		}
		
		int minDistAgentBox = Math.abs(rowBestBox - n.agentRow) + Math.abs(colBestBox - n.agentCol);
		
		//System.err.println(minDistBox);
		return minDistAgentBox;
		
		//H is sum of distance from agent to box and from box to goal
		*/
		
		
		
		
		// Distance from all goals to their closest box
		int tot_dist = 0;
		int minDist = rows * cols;
		for (int ii = 0; ii < goals.size(); ii++) {
				int i = goals.get(ii)[0];
				int j = goals.get(ii)[1];
				char goal = n.goals[i][j];
				if ('a' <= goal && goal <= 'z') {
					//A goal has been found, find all the boxes belonging to this goal.
					minDist = rows * cols;
					for (int k = 0; k < rows; k++) {
						for (int k2 = 0; k2 < cols; k2++) {
							if (Character.toLowerCase(n.boxes[k][k2]) == goal) {
								//Box is found
								int dist = Math.abs(i - k) + Math.abs(j - k2);
								if(dist < minDist) {
									minDist = dist;
								}
							}
						}
					}
				}
				tot_dist = tot_dist + minDist;
				//System.err.println(tot_dist);
			}
		
		return tot_dist;
		
		
		
	   
	}

	public abstract int f(Node n);

	@Override
	public int compare(Node n1, Node n2) {
		return this.f(n1) - this.f(n2);
	}

	public static class AStar extends Heuristic {
		public AStar(Node initialState) {
			super(initialState);
		}

		@Override
		public int f(Node n) {
			return n.g() + this.h(n);
		}

		@Override
		public String toString() {
			return "A* evaluation";
		}
	}

	public static class WeightedAStar extends Heuristic {
		private int W;

		public WeightedAStar(Node initialState, int W) {
			super(initialState);
			this.W = W;
		}

		@Override
		public int f(Node n) {
			return n.g() + this.W * this.h(n);
		}

		@Override
		public String toString() {
			return String.format("WA*(%d) evaluation", this.W);
		}
	}

	public static class Greedy extends Heuristic {
		public Greedy(Node initialState) {
			super(initialState);
		}

		@Override
		public int f(Node n) {
			return this.h(n);
		}

		@Override
		public String toString() {
			return "Greedy evaluation";
		}
	}
}
