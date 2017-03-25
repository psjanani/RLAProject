import numpy as np

from modules import utils
from modules.core import Preprocessor

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

        for num in reward:
            if self.coop:
                rewards = map(lambda r: r + 1, rewards)
            else: # just the killer gets rewarded (perverse if you ask me)
                rewards[int(num)] += 1

        return rewards

    def get_state(self):
        if self.model_name == 'linear':
            return self.frames.reshape(1, self.history_length * self.frame_size[0] * self.frame_size[1] * self.channels)
        elif self.model_name =='stanford' or self.model_name =='deep' or 'dueling' in self.model_name:
            return np.transpose(self.frames, (3, 0, 1, 2))
        else:
            raise Exception("Unsupported model name: " + self.model_name + ".  Accepts linear or deep.")

    def reset(self):
        self.frames = np.zeros([self.frame_size[0], self.frame_size[1], self.channels, self.history_length])

    def get_config(self):
        return {'history_length': self.history_length}

