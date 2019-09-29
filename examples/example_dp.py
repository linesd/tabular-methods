import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.dynamic_programming import value_iteration
from algorithms.dynamic_programming import policy_iteration
from utils.plots import plot_gridworld

###########################################################
#      Run value/policy iteration on a grid world         #
###########################################################

# specify world parameters
num_cols = 10
num_rows = 10
obstructions = np.array([[0,7],[1,1],[1,2],[1,3],[1,7],[2,1],[2,3],
                         [2,7],[3,1],[3,3],[3,5],[3,6],[3,7],[4,3],
                         [4,5],[5,3],[5,9],[6,3],[6,9],[7,1],[7,6],
                         [7,7],[7,8],[7,9],[8,1],[8,5],[8,6],[9,1]])
bad_states = np.array([[1,9],[4,2],[4,4],[7,5],[8,2],[9,9]])
start_state = np.array([[0,4]])
goal_states = np.array([[0,9],[2,2],[8,7]])

# create model
gw = GridWorld(num_cols, num_rows, start_state=start_state, goal_states=goal_states)
gw.add_obstructions(obstructed_states=obstructions,bad_states=bad_states)
gw.add_rewards(step_reward=-1, goal_reward=10,bad_state_reward=-6)
gw.add_transition_probability(p_good_transition=0.7, bias=0.5)
gw.add_discount(discount=0.9)
model = gw.create_gridworld()

# solve with value iteration and policy iteration
vi_value, vi_policy = value_iteration(model, maxiter=100)
pi_value, pi_policy = policy_iteration(model, maxiter=100)

# plot the results
path = "../doc/imgs/value_iteration.png"
plot_gridworld(model, value_function=vi_value, policy=vi_policy, title="Value iteration", path=path)
path = "../doc/imgs/policy_iteration.png"
plot_gridworld(model, value_function=pi_value, policy=pi_policy, title="Policy iteration", path=path)