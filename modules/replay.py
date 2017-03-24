from modules.core import ReplayMemory, RingBuffer, Sample
import numpy as np

class NaiveReplay(ReplayMemory):
    """
    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, random_sample, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.ringbuffer = RingBuffer(max_size)
        self.random_sample = random_sample

    def append(self, state, action, reward, next_state, terminal):
        self.ringbuffer.append(Sample(state, action, reward, next_state, terminal))

    def end_episode(self, final_state, is_terminal):
        pass

    def sample(self, batch_size, indexes=None):
      if self.random_sample:
        return self.get_random_sample(batch_size, indexes)

      size = self.ringbuffer.get_size()

      return self.ringbuffer.get()[size-batch_size:size]

    def get_random_sample(self, batch_size, indexes=None):
        if indexes == None:
            indexes = np.random.randint(self.ringbuffer.get_size(), size=batch_size)
        batch = []
        for index in indexes:
            batch.append(self.ringbuffer.get_ind(index))
        return batch

    def clear(self):
        self.ringbuffer.clear()
