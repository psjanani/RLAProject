# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random

import numpy as np

class AmazonEnv(Env):
    """Implement the Amazon environment.

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
    PICKUP_REWARD = 0

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

    BOX_DROPOFF_MARK = -1

    def __init__(self, grid_size, num_agents, total_boxes, single_train):
        self.nS = grid_size * grid_size
        self.nA = 4**(num_agents)
        self.action_space = spaces.MultiDiscrete([(0,3)] * num_agents)
        self.observation_space = spaces.Discrete(self.nS)
        self.single_train = single_train
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.box_delivery_pt = (grid_size * grid_size) //2

        self.shelves = np.array([ 9, 10, 11, 37, 38, 39 ])

        self.total_boxes = total_boxes
        self.P = dict()
        self._seed()
        self._reset()

    def linear2sub(self, idx):
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)

        return (row, col)

    def random_box_idx(self):
        idx = np.random.random_integers(0, len(self.shelves) - 1)
        return self.shelves[idx]

    def random_idx(self):
        idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
        while idx==self.box_delivery_pt or np.any(self.shelves == idx):
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
                val = self.agent_channel[r][c]
                if val == 2*target_idx or val==2*target_idx-1:
                    return (r, c)

    def myrange(self):
        return 1 if self.single_train else self.num_agents

    def box_in_transit(self):
        return not np.any(self.s < AmazonEnv.BOX_DROPOFF_MARK)

    def _reset(self):
        """Reset the environment.

        Returns
        -------
        randomly generated state
        initial state
        """
        self.s = np.zeros([self.grid_size, self.grid_size])
        self.agent_channel = [[0] * self.grid_size for _ in xrange(self.grid_size)]
        self.rem_boxes=self.total_boxes
        # random placement of agents first
        (r, c) = self.random_idx()
        #for each agent, it will have an even number with box and odd without,i.e. 2i and 2i-1
        self.agent_channel[int(r)][int(c)] = 2

        for i in range(1, self.myrange()):
            (r, c) = self.random_idx()

            while self.agent_channel[int(r)][int(c)] < 2*i + 1:
                #This while loop checks for places which have been occupied by previous agents
                if self.agent_channel[int(r)][int(c)]==0:
                    self.agent_channel[int(r)][int(c)] = 2 * (i + 1)
                else:
                    (r, c) = self.random_idx()

        r, c = self.linear2sub(self.box_delivery_pt)
        self.s[r][c] = AmazonEnv.BOX_DROPOFF_MARK

        self.box_request()

        return self.linear2sub(self.box_pickup_pt), self.agent_channel

    def box_request(self):
        self.box_pickup_pt = self.random_box_idx()
        r, c = self.linear2sub(self.box_pickup_pt)
        self.s[r][c] = AmazonEnv.BOX_DROPOFF_MARK - 1

    def find_agent_indices(self):
        indices = []
        for i in range(self.myrange()):
            indices.append(self.agent_state((i + 1)))
        return indices

    def new_location(self, agent_pos, action):
        # figure out deltas of action
        deltas = AmazonEnv.ACTION_DELTAS[int(action)]
        new_agent_pos_r = min(max(agent_pos[0] + deltas[0], 0), self.grid_size - 1)
        new_agent_pos_c = min(max(agent_pos[1] + deltas[1], 0), self.grid_size - 1)

        lin_idx = self.sub2linear((new_agent_pos_r, new_agent_pos_c))
        if np.any(self.shelves == lin_idx): ## can't collide with shelf
            return agent_pos, lin_idx == self.box_pickup_pt
        return (new_agent_pos_r, new_agent_pos_c), False

    def resolve_conflicts(self, actions, curr_positions, channel):
        next_positions = []
        didnt_move = []
        pickups = []

        for i in range(len(curr_positions)):
            action = actions[i]
            curr_pos = curr_positions[i]

            next_pos, pickedup = self.new_location(curr_pos, action)
            didnt_move.append(next_pos == curr_pos)
            next_positions.append(next_pos)
            pickups.append(pickedup)

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

        return next_positions, pickups

    def _step(self, action):
        action_str = str("".join(action))
        actions = np.zeros(self.myrange())

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

            if i >= self.myrange():
                break

        new_agent_channel = [[0] * self.grid_size for _ in xrange(self.grid_size)]

        reward = [0] * self.myrange()

        next_positions, pickups = self.resolve_conflicts(actions[0:self.myrange()], curr_agent_pos, 'agent')

        for i in range(len(next_positions)):
            r, c = curr_agent_pos[i]
            new_r, new_c = next_positions[i]
            new_agent_channel[new_r][new_c] = self.agent_channel[r][c]

        dropoff = False
        pickup = False
        for i in range(len(next_positions)):
            r, c = self.linear2sub(self.box_pickup_pt)
            del_r, del_c = self.linear2sub(self.box_delivery_pt)
            new_r, new_c = next_positions[i]

            if pickups[i] and not self.s[r][c] == 0:
                # print ('Box picked up\n')
                new_r, new_c = next_positions[i]
                reward[i] = AmazonEnv.PICKUP_REWARD
                new_agent_channel[new_r][new_c]= new_agent_channel[new_r][new_c] - 1
                self.s[r][c] = 0
                pickup = True

            if new_r == del_r and new_c == del_c and self.s[new_r][new_c] == AmazonEnv.BOX_DROPOFF_MARK and new_agent_channel[new_r][new_c] % 2 == 1:
                # print('Box dropped off\n')
                reward[i] = AmazonEnv.DELIVERY_REWARD

                self.rem_boxes -= 1
                new_agent_channel[new_r][new_c] = new_agent_channel[new_r][new_c] + 1
                dropoff = True

        self.agent_channel = new_agent_channel
        is_terminal = self.rem_boxes == 0
        if not is_terminal and dropoff:
            self.box_request()

        if self.box_in_transit():
            box_loc = (-1,-1)
        else:
            box_loc = self.linear2sub(self.box_pickup_pt)

        return (box_loc, self.agent_channel), reward, is_terminal, 'no debug information provided'

    def _render(self, mode='human', close=False):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                lin_idx = self.sub2linear((r,c))

                if self.s[r][c] < AmazonEnv.BOX_DROPOFF_MARK:
                    print("Box", end=' ')
                elif np.any(self.shelves == lin_idx):
                    print("***", end=' ')
                elif self.agent_channel[r][c] % 2 == 1:
                    print('!!!', end=' ')
                elif self.agent_channel[r][c] > 0:
                    print("Agt", end=' ')
                elif self.s[r][c] == AmazonEnv.BOX_DROPOFF_MARK:
                    print("$$$", end=' ')
                else:
                    print("___", end=' ')
            print ("\n")
        print ("\n")
        return

    def set_single_train(self, single_train):
        self.single_train = single_train

    def _seed(self, seed=None):
        """Set the random seed.

        Parameters
        ----------
        seed: int, None
          Random seed used by numpy.random and random.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
    id='Amazon-v0',
    entry_point='envs.amazon_envs:AmazonEnv',
    kwargs={'grid_size':3,'num_agents':1,'total_boxes':2, 'single_train': False}
)

register(
    id='Amazon-v1',
    entry_point='envs.amazon_envs:AmazonEnv',
    kwargs={'grid_size':5, 'num_agents':2, 'total_boxes': 1, 'single_train':False }
)

register(
    id='Amazon-Single-v1',
    entry_point='envs.amazon_envs:AmazonEnv',
    kwargs={'grid_size':7, 'num_agents':2, 'total_boxes': 2, 'single_train':True } 
)