import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from algorithms.temporal_difference import sarsa

def test_sarsa():
    # set seed
    np.random.seed(1)
    # load the test data
    vp = np.load("../data/test_data/test_sarsa.npy")
    test_q = vp[:,0].reshape(-1,1)
    test_pi = vp[:,1].reshape(-1,1)

    # specify world parameters
    num_cols = 4
    num_rows = 4
    obstacles = np.array([[1, 1],
                          [2, 1],
                          [1, 2]])
    bad_states = np.array([[3,0]])
    start_state = np.array([[0, 0]])
    goal_state = np.array([[3, 3]])

    # create world
    gw = GridWorld(num_cols, num_rows, start_state=start_state, goal_states=goal_state)
    gw.add_obstructions(obstructed_states=obstacles, bad_states=bad_states)
    gw.add_rewards(step_reward=-1, goal_reward=10, bad_state_reward=-6)
    gw.add_transition_probability(p_good_transition=0.8, bias=0.5)
    gw.add_discount(0.9)
    model = gw.create_gridworld()

    # solve with sarsa
    q, pi, _ = sarsa(model, alpha=0.1, epsilon=0.2, maxiter=100, maxeps=1000)

    # test value iteration outputs
    assert np.all(q == test_q)
    assert np.all(pi == test_pi)