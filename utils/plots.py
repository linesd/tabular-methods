import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.helper_functions import create_policy_direction_arrays
import numpy as np

def plot_gridworld(model, value_function=None, policy=None, title=None, path=None):
    """
    Plots the grid world solution.

    Parameters
    ----------
    model : python object
        Holds information about the environment to solve
        such as the reward structure and the transition dynamics.

    value_function : numpy array of shape (N, 1)
        Value function of the environment where N is the number
        of states in the environment.

    policy : numpy array of shape (N, 1)
        Optimal policy of the environment.

    title : string
        Title of the plot. Defaults to None.

    path : string
        Path to save image. Defaults to None.
    """
    fig, ax = plt.subplots()
    # colobar max and min
    vmin = np.min(value_function)
    vmax = np.max(value_function)

    if value_function is not None:
        # reshape and set obstructed states to low value
        val = value_function[:-1, 0].reshape(model.num_rows, model.num_cols)
        index = model.obs_states
        val[index[:, 0], index[:, 1]] = -100
        plt.imshow(val, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Value function")
    else:
        val = np.zeros((model.num_rows, model.num_rows))
        plt.imshow(val, vmin=vmin, vmax=vmax)
        plt.xticks(np.arange(-0.5, model.num_rows+0.5, step=1))
        plt.yticks(np.arange(-0.5, model.num_cols+0.5, step=1))
        plt.grid(which="major")
        plt.colorbar(label="Value function")

    # create start and end patches
    start = patches.Circle(tuple(np.flip(model.start_state[0])), 0.2,linewidth=1,
                           edgecolor='b', facecolor='b', label="Start")
    ax.add_patch(start)

    for i in range(model.goal_states.shape[0]):
        end = patches.RegularPolygon(tuple(np.flip(model.goal_states[i,:])), numVertices=5,
                                     radius=0.25, orientation=np.pi, edgecolor='g',
                                     facecolor='g',label="Goal" if i == 0 else None)
        ax.add_patch(end)

    for i in range(model.obs_states.shape[0]):
        obstructed = patches.Rectangle(tuple(np.flip(model.obs_states[i,:])-0.35), 0.7, 0.7,
                                      linewidth=1, edgecolor='orange', facecolor='orange',
                                       label="Obstructed" if i == 0 else None)
        ax.add_patch(obstructed)

    for i in range(model.bad_states.shape[0]):
        bad = patches.Wedge(tuple(np.flip(model.bad_states[i,:])), 0.3, 20, -20,
                            linewidth=1, edgecolor='r', facecolor='none',
                            label="Bad state" if i == 0 else None)
        ax.add_patch(bad)
    # define the gridworld
    X = np.arange(0, model.num_cols, 1)
    Y = np.arange(0, model.num_rows, 1)

    if policy is not None:
        # define the policy direction arrows
        U, V = create_policy_direction_arrays(model, policy)
        # remove the obstructions and final state arrows
        ra = np.vstack((model.obs_states, model.goal_states))
        U[ra[:,0],ra[:,1]] = np.nan
        V[ra[:,0],ra[:,1]] = np.nan
        plt.quiver(X, Y, U, V, label="Policy")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=3)
    if title is not None:
        plt.title(title, fontdict=None, loc='center')
    if path is not None:
        plt.savefig(path, dpi=300)
    plt.show()