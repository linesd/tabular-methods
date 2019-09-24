import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.helper_functions import create_policy_direction_arrays
import numpy as np

def plot_gridworld(model, value_function, policy):

    fig, ax = plt.subplots(1)
    # colobar max and min
    vmin = np.min(value_function)
    vmax = np.max(value_function)
    # reshape and set obstructed states to low value
    val = value_function[:-1,0].reshape(model.num_rows,model.num_cols)
    index = model.obs_states
    val[index[:, 0], index[:, 1]] = -100
    plt.imshow(val, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Value function")
    # create start and end patches
    start = patches.Circle(tuple(np.flip(model.start_state[0])), 0.2,linewidth=1,
                           edgecolor='g', facecolor='g', label="Start")
    end = patches.RegularPolygon(tuple(np.flip(model.goal_state[0])), numVertices=5,
                                 radius=0.25, orientation=np.pi, edgecolor='r',
                                 facecolor='r',label="Goal")
    for i in range(model.obs_states.shape[0]):
        obstructed = patches.Rectangle(tuple(np.flip(model.obs_states[i,:])-0.35), 0.7, 0.7,
                                      linewidth=1, edgecolor='orange', facecolor='orange',
                                       label="Obstructed" if i == 0 else None)
        ax.add_patch(obstructed)
    ax.add_patch(start)
    ax.add_patch(end)
    # define the gridworld
    X = np.arange(0, model.num_cols, 1)
    Y = np.arange(0, model.num_rows, 1)
    # define the policy direction arrows
    U, V = create_policy_direction_arrays(model, policy)
    # remove the obstructions and final state arrows
    ra = np.vstack((model.obs_states, model.goal_state))
    U[ra[:,0],ra[:,1]] = np.nan
    V[ra[:,0],ra[:,1]] = np.nan
    plt.quiver(X, Y, U, V, label="Policy")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
               fancybox=True, shadow=True, ncol=4)
    plt.show()