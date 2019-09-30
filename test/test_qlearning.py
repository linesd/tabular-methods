import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.temporal_difference import qlearning

def test_qlearning():
    np.random.seed(1)
    # load the test data
    vp = np.load("../data/test_data/test_qlearning.npy")
    test_q = vp[:,0].reshape(-1,1)
    test_pi = vp[:,1].reshape(-1,1)

    # specify world parameters
    num_rows = 4
    num_cols = 4
    obstacles = np.array([[1, 1], [2, 1], [1, 2]])
    bad_states = np.array([[3,0]])
    restart_state = np.array([[2, 2]])
    start_state = np.array([[0, 0]])
    goal_state = np.array([[3, 3]])

    # create world
    gw = GridWorld(num_rows=num_rows,
                   num_cols=num_cols,
                   start_state=start_state,
                   goal_states=goal_state)
    gw.add_obstructions(obstructed_states=obstacles,
                        bad_states=bad_states,
                        restart_states=restart_state)
    gw.add_rewards(step_reward=-1,
                   goal_reward=10,
                   bad_state_reward=-6,
                   restart_state_reward=-10)
    gw.add_transition_probability(p_good_transition=0.8,
                                  bias=0.5)
    gw.add_discount(discount=0.9)
    model = gw.create_gridworld()

    # solve with sarsa
    q, pi, _ = qlearning(model, alpha=0.8, epsilon=0.1, maxiter=100, maxeps=1000)

    # test value iteration outputs
    assert np.all(q == test_q)
    assert np.all(pi == test_pi)
