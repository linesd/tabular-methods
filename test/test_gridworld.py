import sys
sys.path.append("..")
import numpy as np
from env.grid_world import GridWorld
from scipy.io import loadmat

def test_gridworld():
    grid_world = loadmat('../data/test_data/gridworld.mat')['model']

    # specify world parameters
    num_cols = 12
    num_rows = 9
    obstructions = np.array([[8,6],[7,6],[6,6],[5,6],[4,6],[3,6],[3,7],[3,8],[3,9]])
    bad_states = np.array([[2,1]])
    start_state = np.array([[0,1]])
    goal_state = np.array([[7,8]])

    # create model
    gw = GridWorld(num_cols, num_rows, start_state=start_state, goal_states=goal_state)
    gw.add_obstructions(obstructed_states=obstructions,bad_states=bad_states)
    gw.add_rewards(step_reward=-1, goal_reward=10,bad_state_reward=-6)
    gw.add_transition_probability(p_good_transition=0.7, bias=0.5)
    gw.add_discount(0.9)
    model = gw.create_gridworld()

    # run tests
    assert np.all(model.R == grid_world['R'][0][0][:,0].reshape(-1,1))
    assert np.all(model.P[:,:,0] == grid_world['P'][0][0][:,:,0])
    assert np.all(model.P[:,:,1] == grid_world['P'][0][0][:,:,1])
    assert np.all(model.P[:,:,2] == grid_world['P'][0][0][:,:,2])
    assert np.all(model.P[:,:,3] == grid_world['P'][0][0][:,:,3])
