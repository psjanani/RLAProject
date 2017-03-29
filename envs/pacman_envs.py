# coding: utf-8
"""Define the Queue environment from problem 3 here."""

from __future__ import (absolute_import, division, print_function,
						unicode_literals)

from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import random
import numpy as np

from distance_from import smart_move

class PacmanEnv(Env):

	metadata = {'render.modes': ['human']}

	MARKS = {
		'empty': '_',
		'barrier': '*',
		'prey': 'O',
		'predator': 'X'
	}

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

	def fill_in_barrier_mask(self, barriers):
		# mask is 1 where there is a barrier
		self.barrier_mask = [[0]*self.grid_size for _ in xrange(self.grid_size) ]
		
		num_barrier_cells = 0

		for barrier in barriers:
			start_row = int(barrier[1])
			start_col = int(barrier[0])

			width = int(barrier[2])
			height = int(barrier[3])

			for r in range(start_row, start_row + height):
				for c in range(start_col, start_col + width):
					self.barrier_mask[r][c] = 1


			num_barrier_cells += height*width

		return num_barrier_cells

	def __init__(self, barriers, grid_size, num_agents, smart_prey, smart_predator):
		# this is total number of agents (must be even number)
		# half of these agents will be prey and half predators
		if num_agents % 2 != 0:
			raise "Argument Error.  Must have even number of agents"

		self.num_agents = num_agents
		self.smart_prey = smart_prey
		self.smart_predator = smart_predator
		self.num_prey = self.num_predators = int(self.num_agents/2)

		self.grid_size = grid_size

		# mask is 1 where there is a barrier
		self.barrier_mask = [[0]*self.grid_size for _ in xrange(self.grid_size) ]
		num_barrier_cells = self.fill_in_barrier_mask(barriers)

		# free cells available to be roamed
		free_cells = (self.grid_size * self.grid_size) - num_barrier_cells

		# simple probability calculation
		self.nS = free_cells * (free_cells - 1)
		for i in range(2, self.num_agents):
			self.nS += self.nS * (free_cells - i)

		self.nA = 4**self.num_agents

		self.action_space = spaces.MultiDiscrete(
			[(0,4)]*self.num_agents
		)

		self.observation_space = spaces.Discrete(self.nS)
		
		self._seed()
		self._reset()

	def sub2Linear((r, c)):
		return r*self.grid_size + c

	def random_idx(self):
		idx = np.random.random_integers(0, self.grid_size * self.grid_size - 1)
		row = int(idx / self.grid_size)
		col = int(idx % self.grid_size)

		return row, col

	def new_location(self, agent_pos, action):
		# figure out deltas of action
		deltas = PacmanEnv.ACTION_DELTAS[int(action)]

		new_agent_pos_r = min(max(agent_pos[0] + deltas[0], 0), self.grid_size - 1)
		new_agent_pos_c = min(max(agent_pos[1] + deltas[1], 0), self.grid_size - 1)

		if(self.barrier_mask[new_agent_pos_r][new_agent_pos_c] == 0):
			return (new_agent_pos_r, new_agent_pos_c)
		else:
			return agent_pos

	def random_valid_idx(self):
		full_positions = np.add(np.add(self.prey_channel, self.predator_channel), self.barrier_mask)

		(row, col) = self.random_idx()

		while full_positions[row][col] != 0:
			(row, col) = self.random_idx()

		return (row, col)

	def _reset(self):
		"""Reset the environment.

		Returns
		-------
		randomly generated state
		initial state
		"""
		self.num_prey = self.num_predators
		self.predator_channel = [[0]*self.grid_size for _ in xrange(self.grid_size) ]
		self.prey_channel = [[0]*self.grid_size for _ in xrange(self.grid_size) ]

		# random placement of predators first
		for i in range(1, self.num_predators + 1):
			(r,c) = self.random_valid_idx()
			self.predator_channel[r][c] = i

		for i in range(1, self.num_prey + 1):
			(r,c) = self.random_valid_idx()
			self.prey_channel[r][c] = 1

		return self.barrier_mask, self.prey_channel, self.predator_channel

	
	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def agent_state(self, channel, target_idx):
		for r in range(self.grid_size):
			for c in range(self.grid_size):
				val = channel[r][c]
				if val == target_idx:
					return (r, c)

	def predator_state(self, target_idx):
		for r in range(self.grid_size):
			for c in range(self.grid_size):
				val = self.predator_channel[r][c]
				if val == target_idx:
					return (r, c)
	
	def _render(self, mode='human', close=False):
		for r in range(self.grid_size):
			for c in range(self.grid_size):
				if self.barrier_mask[r][c] != 0:
					print(PacmanEnv.MARKS['barrier'], end=' ')
				elif self.prey_channel[r][c] != 0:
					print(PacmanEnv.MARKS['prey'], end=' ')
				elif self.predator_channel[r][c] != 0:
					print(PacmanEnv.MARKS['predator'], end=' ')
				else:
					print(PacmanEnv.MARKS['empty'], end=' ')
			print ("")

		return

	def find_prey_indices(self):
		indices = []
		for r in range(self.grid_size):
			for c in range(self.grid_size):
				if self.prey_channel[r][c] == 1:
					indices.append((r,c))
		return indices

	def find_predator_indices(self):
		indices = []
		for i in range(self.num_predators):
			indices.append(self.predator_state(i + 1))
		return indices

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
		action_str = str("".join(action))
		actions = np.zeros(self.num_agents)

		curr_predator_pos = self.find_predator_indices()
		curr_prey_pos = self.find_prey_indices()

		i = 0
		for action in action_str:
			action = action_str[i]
			try:
				action = int(action)
			except ValueError:
				if self.smart_predator:
					action = smart_move(self.barrier_mask,\
						curr_predator_pos[i], curr_prey_pos, 'closer')[0]
				else:
					action = np.random.randint(4)
			actions[i] = action
			i += 1

		if self.smart_prey:
			smart_actions = smart_move(self.barrier_mask, curr_prey_pos, curr_predator_pos, 'away')
			if np.shape(smart_actions)[0] == 0:
				for i in range(self.num_predators, self.num_agents):
					actions[i] = np.random.randint(4)
			else:
				actions[self.num_predators: self.num_agents] = smart_actions
		else:
			for i in range(self.num_predators, self.num_agents):
				actions[i] = np.random.randint(4)

		new_predator_channel = [[0]*self.grid_size for _ in xrange(self.grid_size) ]
		new_prey_channel = [[0]*self.grid_size for _ in xrange(self.grid_size) ]

		reward = ''

		next_positions = self.resolve_conflicts(actions[0:self.num_predators], curr_predator_pos, 'predator')

		for i in range(len(next_positions)):
			r,c = curr_predator_pos[i]
			new_r, new_c = next_positions[i]
			new_predator_channel[new_r][new_c] = self.predator_channel[r][c]

		next_positions = curr_prey_pos
		
		####
		####
		#### TODO UNCOMMENT LATER
		####
		####
		# next_positions = self.resolve_conflicts(actions[self.num_predators:], curr_prey_pos, 'prey')

		for i in range(len(next_positions)):
			r,c = curr_prey_pos[i]
			new_r, new_c = next_positions[i]

			predator_mark_old = new_predator_channel[r][c]
			predator_mark_new = new_predator_channel[new_r][new_c]
			prev_predator_mark = self.predator_channel[new_r][new_c]

			swapped_places = predator_mark_old > 0 and prev_predator_mark > 0 and predator_mark_old == prev_predator_mark

			landed_same_place = predator_mark_new > 0

			if swapped_places:
				self.num_prey -= 1

				reward += str(predator_mark_old)
			elif landed_same_place:
				self.num_prey -= 1
				reward += str(predator_mark_new)
			else:
				new_prey_channel[new_r][new_c] = self.prey_channel[r][c]

		self.predator_channel = new_predator_channel
		self.prey_channel = new_prey_channel

		is_terminal = self.num_prey == 0

		return [self.barrier_mask,  self.prey_channel, self.predator_channel], reward, is_terminal, 'no debug information provided'


# barrier is to parallel barrier
basic_barriers = [ (2, 1, 8, 4), (0, 6, 8, 3) ]
register(
	id='PacmanEnv-v0',
	entry_point='envs.pacman_envs:PacmanEnv',
	kwargs={'barriers': basic_barriers, 'grid_size':10, 'num_agents':4, 'smart_prey': False, 'smart_predator': False})

register(
	id='PacmanEnvSmartPrey-v0',
	entry_point='envs.pacman_envs:PacmanEnv',
	kwargs={'barriers': basic_barriers, 'grid_size':10, 'num_agents':4, 'smart_prey': True, 'smart_predator': False})

register(
	id='PacmanEnvSmartPredators-v0',
	entry_point='envs.pacman_envs:PacmanEnv',
	kwargs={'barriers': basic_barriers, 'grid_size':10, 'num_agents':4, 'smart_prey': False, 'smart_predator': True})

register(
	id='PacmanEnvSmartBoth-v0',
	entry_point='envs.pacman_envs:PacmanEnv',
	kwargs={'barriers': basic_barriers, 'grid_size':10, 'num_agents':4, 'smart_prey': True, 'smart_predator': True})
