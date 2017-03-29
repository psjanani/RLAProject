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

    def __init__(self, frame_size, model_name, num_pred, coop, history_length=1):
        self.history_length = history_length
        self.model_name = model_name
        self.coop = coop
        self.num_pred = num_pred
        self.frame_size = frame_size
        self.model_name = model_name
        self.reset()

    def add_state(self, state):
        state = np.array(state)
        prey_channel = state[1][:][:]
        prey_idxs = np.nonzero(prey_channel)
        prey_channel[prey_idxs] = -2

        state = np.add(np.add(state[0, :, :], state[1, :, :]), state[2, :, :])

        self.frames = state

        return self.frames

    def process_reward(self, reward):
        rewards = [0] * self.num_pred

        reward_val = 10

        for num in reward:
            if self.coop:
                rewards = map(lambda r: r + reward_val, rewards)
            else: # just the killer gets rewarded (perverse if you ask me)
                rewards[int(num) - 1] += reward_val

        return rewards

    def get_state(self, id=None):
        # if id is passed only create state for that particular agent
        # break predator state in last channel into self and others (normalize all to 1s)
        full_frames = np.zeros([self.num_pred, self.frame_size[0], self.frame_size[1]])

        pred_idxs = np.nonzero(self.frames > 0) # [(4, 1), ( 9, 2 )]

        nz_ids = self.frames[pred_idxs] # [ 2, 1] 2 is 2nd predator...


        self.frames[pred_idxs] = 1

        for i in range(self.num_pred):
            my_frame = np.copy(self.frames)
            predator_id = int(nz_ids[i])

            r = pred_idxs[0][i]
            c = pred_idxs[1][i]

            my_frame[r, c] = 2
            full_frames[predator_id - 1, :, :] = my_frame

        if self.model_name == 'linear':
            return full_frame.reshape(self.num_pred, self.frame_size[0] * self.frame_size[1])

        if not id is None:
            return np.expand_dims(np.expand_dims(full_frames[id], axis=-1), axis=0)

        return np.expand_dims(np.expand_dims(full_frames, axis=-1), axis=1)

    def reset(self):
        self.frames = np.zeros([self.frame_size[0], self.frame_size[1]])

    def get_config(self):
        return {'history_length': self.history_length}

