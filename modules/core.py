"""Core classes."""

class RingBuffer:
    """ class that implements a not-yet-full buffer
        Referred from
    https://www.safaribooksonline.com/library/view/python-cookbook\
    /0596001673/ch05s19.html"""
    def __init__(self,size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[int(self.cur)] = x
            self.cur = (int(self.cur) + 1) % self.max

        def get(self):
            """ return list of elements in correct order """
            return self.data[int(self.cur):] + self.data[:int(self.cur)]

        def get_size(self):
            return len(self.data)

        def get_ind(self, ind):
            """ Return a list of elements from the oldest to the newest. """
            return self.data[ind]

    def append(self,x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

    def get_size(self):
        return len(self.data)

    def get_ind(self, ind):
        """ Return a list of elements from the oldest to the newest. """
        return self.data[ind]

    def clear(self):
        self.data = []


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, state, action, reward, next_state, is_terminal = False):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_terminal = is_terminal
