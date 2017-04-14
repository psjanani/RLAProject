# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random

import numpy as np


class WarehouseEnv(Env):
    """Implement the Warehouse environment.

    Parameters
    ----------

    Attributes
    ----------
    nS: number of states
    nA: number of actions
    P: environment model
    """
    metadata = {'render.modes': ['human']}



    DELIVERY_REWARD = 1

    EMPTY_MARK = 0
    BOX_DELIVERY_MARK = 1
    DELIVERY_MARK = 2
    AGENT_MARK = 3
    BOX_MARK = 4
    AGENT_W_BOX = 5

    ACTION_DELTAS = [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1]
    ]

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self,grid_size,num_agents):
        self.nS = grid_size * grid_size
        self.nA = 4
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.grid_size=grid_size
        self.num_agents=num_agents
        self.box_delivery_pt=grid_size * grid_size - 1
        self.P = dict()
        self._seed()
        self._reset()

    def linear2sub(self, idx):
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)

        return (row, col)

    def linear_insert(self, obj, linear_idx, val):
        (row, col) = self.linear2sub(linear_idx)
        self[obj][row][col] = val

    def sub2Linear((r, c)):
        return r*self.grid_size + c

    def _reset(self):
        """Reset the environment.

        Returns
        -------
        randomly generated state
        initial state
        """
        # self.s = np.zeros(WarehouseEnv.GRID_SIZE, WarehouseEnv.GRID_SIZE)
        self.s = [[WarehouseEnv.EMPTY_MARK]*self.grid_size for _ in xrange(self.grid_size) ]

        # find random position for agent
        starting_agent_pos =self.observation_space.sample()

        # find random position for box not already occupied
        starting_box_pos = self.observation_space.sample()

        while starting_box_pos == 0 or starting_box_pos == starting_agent_pos:
            starting_box_pos = self.observation_space.sample()

        # insert them into state using linear indexing
        r,c = self.linear2sub(starting_agent_pos)
        self.s[r][c] = WarehouseEnv.AGENT_MARK

        r,c = self.linear2sub(starting_box_pos)
        self.s[r][c] = WarehouseEnv.BOX_MARK

        r, c = self.linear2sub(self.box_delivery_pt)
        self.s[r][c] = self.box_delivery_pt

        return self.s

    def agent_state(self, s):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = s[r][c]
                if val == WarehouseEnv.AGENT_MARK:
                    return [False, (r, c)]

                if val == WarehouseEnv.AGENT_W_BOX:
                    return [True, (r, c)]

    def new_location(self, agent_pos, action):
        # figure out deltas of action
        deltas = WarehouseEnv.ACTION_DELTAS[action]

        new_agent_pos_r = min(max(agent_pos[0] + deltas[0], 0), self.grid_size - 1)
        new_agent_pos_c = min(max(agent_pos[1] + deltas[1], 0), self.grid_size - 1)

        return (new_agent_pos_r, new_agent_pos_c)

    def _step(self, action):
        """Execute the specified action

        Parameters
        ----------
        action: int
          A number in range [0, 3]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """

        agent_state = self.agent_state(self.s)

        # get current agent state
        carrying_box = agent_state[0]
        old_agent_pos = agent_state[1]
        r_old,c_old = old_agent_pos

        # find new position
        new_agent_pos = self.new_location(old_agent_pos, action)
        r_new,c_new = new_agent_pos

        reward = 0
        is_terminal=False

        dest_val = self.s[r_new][c_new]

        if dest_val == WarehouseEnv.BOX_DELIVERY_MARK:
            if carrying_box:
                reward = WarehouseEnv.DELIVERY_REWARD
                is_terminal = True
                self.s[r_new][c_new] = WarehouseEnv.AGENT_MARK
            else:
                # can't go to that state without the GOODS
                self.s[r_old][c_old] = WarehouseEnv.AGENT_MARK
                new_agent_pos = old_agent_pos
        elif dest_val == WarehouseEnv.BOX_MARK:
            self.s[r_new][c_new] = WarehouseEnv.AGENT_W_BOX
        else:
            self.s[r_new][c_new] = self.s[r_old][c_old]

        # set previous
        if new_agent_pos != old_agent_pos:
            self.s[r_old][c_old] = WarehouseEnv.EMPTY_MARK

        return self.s, reward, is_terminal, 'No debug info provided'

    def _render(self, mode='human', close=False):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                v = self.s[r][c]

                if v == WarehouseEnv.BOX_MARK:
                    print("Box", end=' ')
                elif v == WarehouseEnv.AGENT_MARK:
                    print("Agt", end=' ')
                elif v == WarehouseEnv.BOX_DELIVERY_MARK:
                    print('Del', end=' ')
                elif v == WarehouseEnv.AGENT_W_BOX:
                    print('!!!', end=' ')
                else:
                    print("___", end=' ')
            print ("")
        print ("\n")
        return

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def query_model(self, state, action):
        """Return the possible transition outcomes for a state-action pair.

        This should be in the same format at the provided environments
        in section 2.

        Parameters
        ----------
        state
          State used in query. Should be in the same format at
          the states returned by reset and step.
        action: int
          The action used in query.

        Returns
        -------
        [(prob, nextstate, reward, is_terminal), ...]
          List of possible outcomes
        """

        agent_state = self.agent_state(state)

        # get current agent state
        carrying_box = agent_state[0]
        agent_pos = agent_state[1]

        # update state now that it's no longer there
        state[agent_pos] = WarehouseEnv.EMPTY_MARK

        # find new position
        new_agent_pos = self.new_location(old_agent_pos, action)

        reward = 0
        is_terminal=False

        if new_agent_pos == DELIVERY_MARK:
            if carrying_box:
                reward = WarehouseEnv.DELIVERY_REWARD
                is_terminal = True
                state[new_agent_pos] = WarehouseEnv.AGENT_MARK
            else:
                # can't go to that state without the GOODS
                state[agent_pos] = WarehouseEnv.AGENT_MARK
        elif new_agent_pos == WarehouseEnv.BOX_MARK:
            state[new_agent_pos] = WarehouseEnv.AGENT_W_BOX
        else:
            state[new_agent_pos] = WarehouseEnv.AGENT_MARK

        return [(1.0, state, reward, is_terminal)]

    def get_action_name(self, action):
        if action == QueueEnv.UP:
            return 'up'
        elif action == QueueEnv.RIGHT:
            return 'right'
        elif action == QueueEnv.DOWN:
            return 'down'
        elif action == QueueEnv.LEFT:
            return 'left'
        return 'UNKNOWN'


register(
    id='Warehouse-v0',
    entry_point='envs.warehouse_envs:WarehouseEnv',
    kwargs={'grid_size':10,'num_agents':4})
