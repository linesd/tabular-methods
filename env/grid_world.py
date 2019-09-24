import numpy as np
from utils.helper_functions import row_col_to_seq
from utils.helper_functions import seq_to_col_row

class GridWorld:

    def __init__(self, num_cols, num_rows, start_state, goal_states):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.start_state = start_state
        self.goal_states = goal_states
        self.obs_states = None
        self.bad_states = None
        self.num_bad_states = 0
        self.p_good_trans = None
        self.bias = None
        self.r_step = None
        self.r_goal = None
        self.r_dead = None
        self.gamma = 1 # default is no discounting

    def add_obstructions(self, obstructed_states=None, bad_states=None):
        self.obs_states = obstructed_states
        self.bad_states = bad_states
        if bad_states is not None:
            self.num_bad_states = bad_states.shape[0]

    def add_transition_probability(self, p_good_transition, bias):
        self.p_good_trans = p_good_transition
        self.bias = bias

    def add_rewards(self, step_reward, goal_reward, bad_state_reward=None):
        self.r_step = step_reward
        self.r_goal = goal_reward
        self.r_bad = bad_state_reward

    def add_discount(self, discount):
        self.gamma = discount

    def create_gridworld(self):
        NUM_ACTIONS = 4
        self.num_states = self.num_cols * self.num_rows + 1
        self.start_state_seq = row_col_to_seq(self.start_state, self.num_cols)
        self.goal_states_seq = row_col_to_seq(self.goal_states, self.num_cols)

        # rewards structure
        self.R = self.r_step * np.ones((self.num_states, 1))
        self.R[self.num_states-1] = 0
        self.R[self.goal_states_seq] = self.r_goal
        for i in range(self.num_bad_states):
            if self.r_bad is None:
                raise Exception("Bad state specified but no reward is given")
            bad_state = row_col_to_seq(self.bad_states[i,:].reshape(1,-1), self.num_cols)
            self.R[bad_state, :] = self.r_bad

        # probability model
        if self.p_good_trans == None:
            raise Exception("Must assign probability and bias terms via the add_transition_probability method.")

        self.P = np.zeros((self.num_states,self.num_states,NUM_ACTIONS))
        for action in range(NUM_ACTIONS):
            for state in range(self.num_states):

                # check if state is the fictional end state - self transition
                if state == self.num_states-1:
                    self.P[state, state, action] = 1
                    continue

                # check if the state is the goal state or an obstructed state - transition to end
                row_col = seq_to_col_row(state, self.num_cols)
                end_states = np.vstack((self.obs_states, self.goal_states))
                if any(np.sum(np.abs(end_states-row_col), 1) == 0):
                    self.P[state, self.num_states-1, action] = 1

                # else consider stochastic effects of action
                else:
                    for dir in range(-1,2,1):
                        direction = self._get_direction(action, dir)
                        next_state = self._get_state(state, direction)
                        if dir == 0:
                            prob = self.p_good_trans
                        elif dir == -1:
                            prob = (1 - self.p_good_trans)*(self.bias)
                        elif dir == 1:
                            prob = (1 - self.p_good_trans)*(1-self.bias)

                        self.P[state, next_state, action] += prob

        return self

    def _get_direction(self, action, direction):
        left = [2,3,1,0]
        right = [3,2,0,1]
        if direction == 0:
            new_direction = action
        elif direction == -1:
            new_direction = left[action]
        elif direction == 1:
            new_direction = right[action]
        else:
            raise Exception("getDir received an unspecified case")
        return new_direction

    def _get_state(self, state, direction):
        row_change = [-1,1,0,0]
        col_change = [0,0,-1,1]
        row_col = seq_to_col_row(state, self.num_cols)
        row_col[0,0] += row_change[direction]
        row_col[0,1] += col_change[direction]

        # check for invalid states
        if (np.any(row_col < 0) or
            np.any(row_col[:,0] > self.num_rows-1) or
            np.any(row_col[:,1] > self.num_cols-1) or
            np.any(np.sum(abs(self.obs_states - row_col), 1)==0)):
            state_out = state
        else:
            state_out = row_col_to_seq(row_col, self.num_cols)[0]

        return state_out