import numpy as np
from env.grid_world import GridWorld
import matplotlib.pyplot as plt
from scipy.io import loadmat

def test_smallworld():
    small_world = loadmat('../data/test_data/smallworld.mat')['model']

    num_cols = 4
    num_rows = 4
    obstacles = np.array([[1,1],[2,1],[1,2]])
    start_state = np.array([[0,0]])
    goal_state = np.array([[3,3]])

    gw = GridWorld(num_cols, num_rows, start_state=start_state, goal_state=goal_state)
    gw.add_obstructions(obstacles)
    gw.add_rewards(step_reward=-1, goal_reward=10)
    gw.add_transition_probability(p_good_transition=0.8, bias=0.5)
    gw.add_discount(0.9)
    model = gw.create_gridworld()


    assert np.all(model.R == small_world['R'][0][0][:,0].reshape(-1,1))
    assert np.all(model.P[:,:,0] == small_world['P'][0][0][:,:,0])
    assert np.all(model.P[:,:,1] == small_world['P'][0][0][:,:,1])
    assert np.all(model.P[:,:,2] == small_world['P'][0][0][:,:,2])
    assert np.all(model.P[:,:,3] == small_world['P'][0][0][:,:,3])
