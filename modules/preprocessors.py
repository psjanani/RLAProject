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

    def __init__(self, frame_size, model_name, num_pred, coop, channels, history_length=1):
        self.history_length = history_length
        self.channels = channels
        self.model_name = model_name
        self.coop = coop
        self.num_pred = num_pred
        self.frame_size = frame_size
        self.model_name = model_name
        self.reset()

    def add_state(self, state):
        state = np.transpose(state, (1, 2, 0))

        self.frames[:,:,:,:self.history_length - 1] = self.frames[:,:,:,1:]

        self.frames[:,:,:,-1] = state

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

    def get_state(self, id=None): # if id is passed only create state for that particular agent
        # break predator state in last channel into self and others (normalize all to 1s)
        pred_channel = self.frames[:, :, -1, 0] ## only supports history_length to be 1 for now
        decoupled_states = np.zeros([self.num_pred, self.frame_size[0], self.frame_size[1], self.channels])

        nzs = np.nonzero(pred_channel) # [(4, 1), ( 9, 2 )]

        nz_ids = pred_channel[nzs] # [ 2, 1] 2 is 2nd predator...
        pred_channel[nzs] = 1

        for i in range(self.num_pred):

            for iter1 in range(self.channels - 1):
                for r in range(self.frame_size[0]):
                    for c in range(self.frame_size[1]):
                        decoupled_states[i, r, c, iter1] = np.copy(self.frames[r, c, iter1, 0])

            predator_idx = int(nz_ids[i])

            r = nzs[0][i]
            c = nzs[1][i]

            decoupled_states[predator_idx - 1, r, c, -2] = 0
            decoupled_states[predator_idx - 1, r, c, -1] = 1

        if self.model_name == 'linear':
            decoupled_states = decoupled_states.reshape(self.num_pred, self.history_length * self.frame_size[0] * self.frame_size[1] * (self.channels))

        if not id is None:
            return np.expand_dims(decoupled_states[id], axis=0)

        return np.expand_dims(decoupled_states, axis=1)


    def reset(self):
        self.frames = np.zeros([self.frame_size[0], self.frame_size[1], self.channels - 1, self.history_length])

    def get_config(self):
        return {'history_length': self.history_length}

