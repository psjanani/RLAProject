import numpy as np

from modules import utils
from modules.core import Preprocessor

from PIL import Image

class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, frame_size, model_name, num_pred, coop, is_amazon, history_length=1):
        self.history_length = history_length
        self.model_name = model_name
        self.coop = coop
        self.num_pred = num_pred
        self.frame_size = frame_size
        self.model_name = model_name

        self.is_amazon = is_amazon
        self.reset()

    def add_state(self, state):
        if self.is_amazon:
            return self.add_amazon_state(state)
        else:
            return self.add_pacman_state(state)

    def add_amazon_state(self, state):
        state = np.array(state)
        self.frames = state
        return self.frames

    def add_pacman_state(self, state):
        state = np.array(state)
        prey_channel = state[1][:][:]
        prey_idxs = np.nonzero(prey_channel)
        prey_channel[prey_idxs] = -2

        state = np.add(np.add(state[0, :, :], state[1, :, :]), state[2, :, :])

        self.frames = state

        return self.frames

    def process_reward(self, reward):
        rewards = [0] * self.num_pred

        reward_val = 1

        for num in reward:
            if self.coop:
                rewards = map(lambda r: r + reward_val, rewards)
            else: # just the killer gets rewarded (perverse if you ask me)
                rewards[int(num) - 1] += reward_val

        return rewards

    def get_state(self, id=None):
        if self.is_amazon:
            return self.get_amazon_state(id)
        else:
            return self.get_pacman_state(id)

    def get_amazon_state(self, id):
        full_frames = np.zeros([self.num_pred, self.frame_size[0], self.frame_size[1]])
        agent_locs = np.nonzero(self.frames > 0)

        nz_ids = self.frames[agent_locs]

        def is_odd(num):
            return not num % 2 == 0

        for i in range(self.num_pred):
            my_frame = np.copy(self.frames)

            predator_val = int(nz_ids[i])

            my_loc = agent_locs[i]

            my_r = agent_locs[0][i]
            my_c = agent_locs[1][i]

            other_idx = 1 if i == 0 else 0

            other_r = agent_locs[0][other_idx]
            other_c = agent_locs[1][other_idx]

            if is_odd(predator_val):
                predator_id = (predator_val + 1)/2
            else:
                predator_id = predator_val/2

            if predator_id == 1:
                my_frame[my_r, my_c] += 2 #downgrade other agent
                my_frame[other_r, other_c] -= 2

            full_frames[predator_id - 1, :, :] = my_frame

        full_frames = np.divide(full_frames, 4.0)

        if not id is None:
            return np.expand_dims(np.expand_dims(full_frames[id], axis=-1), axis=0)

        return np.expand_dims(np.expand_dims(full_frames, axis=-1), axis=1)

    def get_pacman_state(self, id):
        # if id is passed only create state for that particular agent
        # break predator state in last channel into self and others (normalize all to 1s)
        full_frames = np.zeros([self.num_pred, self.frame_size[0], self.frame_size[1]])

        pred_idxs = np.nonzero(self.frames > 0) # [(4, 1), ( 9, 2 )]

        nz_ids = self.frames[pred_idxs] # [ 2, 1] 2 is 2nd predator...

        for i in range(self.num_pred):
            my_frame = np.copy(self.frames)
            my_frame[pred_idxs] = 1

            predator_id = int(nz_ids[i])

            r = pred_idxs[0][i]
            c = pred_idxs[1][i]

            my_frame[r, c] = 2
            full_frames[predator_id - 1, :, :] = my_frame

        full_frames += 2
        
        full_frames = np.divide(full_frames, 4)

        if self.model_name == 'linear':
            return full_frames.reshape(self.num_pred, self.frame_size[0] * self.frame_size[1])

        if not id is None:
            return np.expand_dims(np.expand_dims(full_frames[id], axis=-1), axis=0)

        return np.expand_dims(np.expand_dims(full_frames, axis=-1), axis=1)

    def reset(self):
        self.frames = np.zeros([self.frame_size[0], self.frame_size[1]])

    def get_config(self):
        return {'history_length': self.history_length}

