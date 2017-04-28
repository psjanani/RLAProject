# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random

import numpy as np
import PIL
from PIL import Image
from pylab import figure, axes, pie, title, show
from distance_from import smart_move
import matplotlib
from matplotlib import cm
from PIL import ImageEnhance
import subprocess
import shutil

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



    DELIVERY_REWARD = 2
    PICKUP_REWARD=1

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

    def __init__(self,grid_size,num_agents,total_boxes):
        self.nS = grid_size * grid_size
        self.nA = 4**(num_agents)
        self.action_space = spaces.MultiDiscrete([(0,3)]*num_agents)
        self.observation_space = spaces.Discrete(self.nS)
        self.grid_size=grid_size
        self.num_agents=num_agents
        self.box_delivery_pt=grid_size * grid_size - 1
        self.box_pickup_pt=0
        self.total_boxes=total_boxes
        self.P = dict()
        self._seed()
        self._reset()

    def linear2sub(self, idx):
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)

        return (row, col)

    def random_idx(self):
        idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
        while idx==self.box_delivery_pt or idx==self.box_pickup_pt:
            idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
            #print(idx)
        row = int(idx / self.grid_size)
        col = int(idx % self.grid_size)
        #print (row,col)
        return row, col


    def linear_insert(self, obj, linear_idx, val):
        (row, col) = self.linear2sub(linear_idx)
        self[obj][row][col] = val

    def sub2Linear((r, c)):
        return r*self.grid_size + c

    def agent_state(self, target_idx):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.agent_channel[r][c]
                if val == 2*target_idx or val==2*target_idx-1:
                    return (r, c)

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

        for i in range(1, self.num_agents):
            (r, c) = self.random_idx()
            #print(r, c)
            #print(self.agent_channel[int(r)][int(c)])
            while self.agent_channel[int(r)][int(c)]<2*i+1 :
                #This while loop checks for places which have been occupied by previous agents
                if self.agent_channel[int(r)][int(c)]==0:
                    self.agent_channel[int(r)][int(c)] = 2*(i+1)
                else:
                    (r, c) = self.random_idx()

        #print (self.agent_channel)
        # find position for box not already occupied
        starting_box_pos = self.box_pickup_pt

        r,c = self.linear2sub(starting_box_pos)
        self.s[r][c] = 1

        r, c = self.linear2sub(self.box_delivery_pt)
        self.s[r][c] = 2

        return self.s

    def box_request(self):
        r, c = self.linear2sub(self.box_pickup_pt)
        self.s[r][c] = 1

    def find_agent_indices(self):
        indices = []
        for i in range(self.num_agents):
            indices.append(self.agent_state((i + 1)))
        return indices

    def new_location(self, agent_pos, action):
        # figure out deltas of action
        #print (action)
        deltas = WarehouseEnv.ACTION_DELTAS[int(action)]
        #print (agent_pos)
        new_agent_pos_r = min(max(agent_pos[0] + deltas[0], 0), self.grid_size - 1)
        new_agent_pos_c = min(max(agent_pos[1] + deltas[1], 0), self.grid_size - 1)
        #print (new_agent_pos_r,new_agent_pos_c)
        return (new_agent_pos_r, new_agent_pos_c)

    def resolve_conflicts(self, actions, curr_positions, channel):
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
        action_str = action
        # print (action_str)
        actions = np.zeros(self.num_agents)

        curr_agent_pos = self.find_agent_indices()
        #print (curr_agent_pos)
        i = 0
        for action in action_str:
            action = action_str[i]
            # print (action)
            try:
                action = int(action)
            except ValueError:
                action = np.random.randint(4)
            # print (action)
            actions[i] = action
            i += 1


        new_agent_channel = [[0] * self.grid_size for _ in xrange(self.grid_size)]

        reward = ''

        next_positions = self.resolve_conflicts(actions[0:self.num_agents], curr_agent_pos, 'agent')

        for i in range(len(next_positions)):
            r, c = curr_agent_pos[i]
            new_r, new_c = next_positions[i]
            new_agent_channel[new_r][new_c] = self.agent_channel[r][c]

        #print (next_positions)
        for i in range(len(next_positions)):
            r, c = self.linear2sub(self.box_pickup_pt)
            del_r, del_c = self.linear2sub(self.box_delivery_pt)
            new_r, new_c = next_positions[i]

            if new_r==r and new_c==c and self.s[r][c]==1 and new_agent_channel[new_r][new_c]%2==0:
                print ('Box picked up\n')
                new_r, new_c = next_positions[i]
                reward = reward + str(WarehouseEnv.PICKUP_REWARD)
                new_agent_channel[new_r][new_c]=new_agent_channel[new_r][new_c]-1
                self.s[r][c]=0

            if new_r==del_r and new_c==del_c and self.s[new_r][new_c]==2 and new_agent_channel[new_r][new_c]%2==1:
                print('Box dropped off\n')
                reward=reward+str(WarehouseEnv.DELIVERY_REWARD)
                self.rem_boxes=self.rem_boxes-1
                new_agent_channel[new_r][new_c] = new_agent_channel[new_r][new_c] +1
                self.box_request()

        is_terminal = self.rem_boxes == 0
        #print (new_agent_channel)
        self.agent_channel=new_agent_channel
        return self.agent_channel, reward, is_terminal, 'no debug information provided'


    def render(self, iter,mode='human', close=False):

        image = np.zeros((self.grid_size, self.grid_size))
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.s[r][c] == 1:
                    image[r][c] = 0.0
                elif self.agent_channel[r][c] % 2 == 0 and self.agent_channel[r][c] > 0:
                    image[r][c] = 0.5
                elif self.agent_channel[r][c] % 2 == 1:
                    image[r][c] = 0.2
                else:
                    image[r][c] = 1.0

        basewidth = 1000
        wpercent = (basewidth / self.grid_size)
        hsize = int((self.grid_size) * float(wpercent))
        # print (barrier)
        image = Image.fromarray(np.uint8(cm.gist_earth(image) * 255))
        iterstr = '00' + str(iter)
        iterstr = iterstr[len(iterstr) - 3:len(iterstr)]
        image = image.resize((basewidth, hsize))
        enhancer = ImageEnhance.Sharpness(image)
        factor = 4.0
        image = enhancer.enhance(factor)
        image.save('images/env' + iterstr + '.png')

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.s[r][c] == 1:
                    print("Box", end=' ')
                elif self.agent_channel[r][c]%2==0 and self.agent_channel[r][c]>0:
                    print("Agt", end=' ')
                elif self.agent_channel[r][c]%2==1:
                    print('!!!', end=' ')
                else:
                    print("___", end=' ')
            print ("")
        print ("\n")
        if iter%50==0:
            subprocess.call(['ffmpeg', '-f', 'image2', '-r', '3', '-i', 'images/env%03d.png', '-vcodec', 'mpeg4', '-y',
                             'movie.mp4'])


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
    kwargs={'grid_size':4,'num_agents':2,'total_boxes':10})
