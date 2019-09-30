import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.temporal_difference import sarsa
from utils.plots import plot_gridworld
np.random.seed(2)

###########################################################
#               Run SARSA on cliff walk                   #
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

# solve with SARSA
q_function, pi, state_counts = sarsa(model, alpha=0.1, epsilon=0.2, maxiter=100, maxeps=100000)

# plot the results
path = "../doc/imgs/sarsa_cliffworld.png"
plot_gridworld(model, policy=pi, state_counts=state_counts, title="SARSA", path=path)
