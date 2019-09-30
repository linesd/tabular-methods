import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.temporal_difference import sarsa
from algorithms.temporal_difference import qlearning
from utils.plots import plot_gridworld
np.random.seed(2)

###########################################################
#        Run SARSA / Q-Learning on cliff world            #
###########################################################

# specify world parameters
num_rows = 4
num_cols = 12
restart_states = np.array([[3,1],[3,2],[3,3],[3,4],[3,5],
                           [3,6],[3,7],[3,8],[3,9],[3,10]])
start_state = np.array([[3,0]])
goal_states = np.array([[3,11]])

# create model
gw = GridWorld(num_rows=num_rows,
               num_cols=num_cols,
               start_state=start_state,
               goal_states=goal_states)
gw.add_obstructions(restart_states=restart_states)
gw.add_rewards(step_reward=-1,
               goal_reward=10,
               restart_state_reward=-100)
gw.add_transition_probability(p_good_transition=1,
                              bias=0)
gw.add_discount(discount=0.9)
model = gw.create_gridworld()

# solve with value iteration and policy iteration
q_function, pi = sarsa(model, alpha=0.1, epsilon=0.2, maxiter=100, maxeps=100000)

# plot the results
path = "../doc/imgs/sarsa_cliffworld.png"
plot_gridworld(model, value_function=q_function, policy=pi, title="SARSA", path=path)
