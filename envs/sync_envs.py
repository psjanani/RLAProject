# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random

import numpy as np

class SyncEnv(Env):
    """Implement the Sync environment.

    Parameters
    ----------

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}

    REWARD = 100
    PENALTY = 0

    ACTION_DELTAS = [
        [0, 0],
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1]
    ]

    NO = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

    def __init__(self, grid_size):
        self.num_agents = 2
        self.nS = grid_size * grid_size
        self.nA = 5**(self.num_agents)
        self.action_space = spaces.MultiDiscrete([(0,3)] * self.num_agents)
        self.observation_space = spaces.Discrete(self.nS)
        self.grid_size = grid_size

        self.sync_marks = [ 0, (grid_size * grid_size) - 1]

        self.P = dict()
        self._seed()
        self._reset()

    def linear2sub(self, idx):
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)

        return (row, col)

    def random_idx(self):
        idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
        while idx in self.sync_marks:
            idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)
        return row, col

    def linear_insert(self, obj, linear_idx, val):
        (row, col) = self.linear2sub(linear_idx)
        self[obj][row][col] = val

    def sub2linear(self, (r, c)):
        return r*self.grid_size + c

    def agent_state(self, target_idx):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.agent_channel[r][c] == target_idx:
                    return (r, c)

    def _reset(self):
        self.agent_channel = [[0] * self.grid_size for _ in xrange(self.grid_size)]

        # random placement of agents first
        (r, c) = self.random_idx()
        self.agent_channel[int(r)][int(c)] = 1

        (r, c) = self.random_idx()
        while self.agent_channel[int(r)][int(c)] == 1:
            (r, c) = self.random_idx()

        self.agent_channel[int(r)][int(c)] = 2

        return self.agent_channel

    def find_agent_indices(self):
        indices = []
        for i in range(self.num_agents):
            indices.append(self.agent_state((i + 1)))
        return indices

    def new_location(self, agent_pos, action):
        # figure out deltas of action
        deltas = SyncEnv.ACTION_DELTAS[int(action)]
        new_agent_pos_r = min(max(agent_pos[0] + deltas[0], 0), self.grid_size - 1)
        new_agent_pos_c = min(max(agent_pos[1] + deltas[1], 0), self.grid_size - 1)

        lin_idx = self.sub2linear((new_agent_pos_r, new_agent_pos_c))
        return new_agent_pos_r, new_agent_pos_c

    def resolve_conflicts(self, actions, curr_positions):
        next_positions = []
        didnt_move = []

        for i in range(len(curr_positions)):
            action = actions[i]
            curr_pos = curr_positions[i]

            next_pos = self.new_location(curr_pos, action)
            didnt_move.append(next_pos == curr_pos)
            next_positions.append(next_pos)

        pos_counters = {}
        for i in range(len(next_positions)):
            next_pos = next_positions[i]

            if next_pos not in pos_counters:
                pos_counters[next_pos] = [i]
            else:
                listt = pos_counters[next_pos]
                listt.append(i)

        for pos in pos_counters:
            if len(pos_counters[pos]) > 1:
                for idx in pos_counters[pos]:
                    if not didnt_move[idx]:
                        next_positions[idx] = curr_positions[idx]

        return next_positions

    def _step(self, action):
        action_str = str("".join(action))
        actions = np.zeros([self.num_agents,])

        curr_agent_pos = self.find_agent_indices()
        i = 0
        for action in action_str:
            action = action_str[i]
            try:
                action = int(action)
            except ValueError:
                action = np.random.randint(4)
            actions[i] = action
            i += 1

        new_agent_channel = [[0] * self.grid_size for _ in xrange(self.grid_size)]
        next_positions = self.resolve_conflicts(actions, curr_agent_pos)
        reached = [ False ] * self.num_agents
        is_terminal = False
        reward = [ 0.0 ] * self.num_agents

        for i in range(len(next_positions)):
            r, c = curr_agent_pos[i]
            new_r, new_c = next_positions[i]
            new_agent_channel[new_r][new_c] = self.agent_channel[r][c]

        for i in range(len(next_positions)):

            new_r, new_c = next_positions[i]

            index = self.sub2linear((new_r, new_c))

            if index in self.sync_marks:
                reward[i] = SyncEnv.PENALTY
                reached[i] = True
                is_terminal = True

        if reached[0] and reached[1]:
            reward[0] = SyncEnv.REWARD
            reward[1] = SyncEnv.REWARD

        # set agent_channel to new_agent_channel
        self.agent_channel = new_agent_channel

        return self.agent_channel, reward, is_terminal, 'no debug information provided'

    def _render(self, mode='human', close=False):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                lin_idx = self.sub2linear((r,c))

                if self.agent_channel[r][c] > 0:
                    print("Agt" + str(self.agent_channel[r][c]), end=' ')
                elif lin_idx in self.sync_marks:
                    print("___", end=' ')
                else:
                    print("***", end=' ')
            print ("\n")
        print ("\n")

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

register(
    id='SyncEnv-v0',
    entry_point='envs.sync_envs:SyncEnv',
    kwargs={ 'grid_size': 5 }
)
