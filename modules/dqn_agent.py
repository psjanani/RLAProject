from keras.optimizers import Adam
import numpy as np
from replay import NaiveReplay
from keras.models import Model
from utils import get_hard_target_model_updates, render
from modules.preprocessors import HistoryPreprocessor
from modules.policy import LinearDecayGreedyEpsilonPolicy, GreedyPolicy,\
	GreedyEpsilonPolicy, UniformRandomPolicy

"""Main DQN agent."""

class DQNAgent:
	"""Class implementing DQN.
	Parameters
	----------
	q_network: keras.models.Model
		Your Q-network model.
	preprocessors: modules.core.Preprocessor
		The preprocessor class. See the associated classes for more
		details.  One or two depending (iff dueling / double Q-Network)
	memory: modules.core.Memory
		Your replay memory.
	gamma: float
		Discount factor.
	target_update_freq: float
		Frequency to update the target network. You can either provide a
		number representing a soft target update (see utils.py) or a
		hard target update (see utils.py and Atari paper.)
	num_burn_in: int
		Before you begin updating the Q-network your replay memory has
		to be filled up with some number of samples. This number says
		how many.
	train_freq: int
		How often you actually update your Q-Network. Sometimes
		stability is improved if you collect a couple samples for your
		replay memory, for every Q-network update that you run.
	batch_size: int
		How many samples in each minibatch.
	"""
	def __init__(self, network, target_network, args):
		self.network = network
		self.network_name = args.network_name
		self.memory = args.memory
		self.gamma = args.gamma
		self.num_actions = int(args.num_actions)
		self.target_update_freq = args.target_update_freq
		self.num_burn_in = int(args.num_burn_in)
		self.eval_freq = args.eval_freq
		self.batch_size = args.batch_size
		self.algorithm = args.algorithm
		self.update_freq = int(args.update_freq)
		self.coin_flip = self.algorithm == 'double' and self.network_name == 'linear'
		# 'basic' algorithm has no target fixing or experience replay
		self.target_fixing = not self.algorithm == 'basic' and not self.coin_flip
		self.verbose = args.verbose > 0

		self.preprocessor = HistoryPreprocessor((args.dim, args.dim), args.network_name, args.channels, args.history)
		self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.end_epsilon, args.num_decay_steps)

		self.buffer = None
		if (args.num_burn_in != 0):
			self.buffer = NaiveReplay(self.memory, not self.algorithm == 'basic', None)
		
		if self.target_fixing:
			self.target = Model.from_config(self.network.get_config())
			get_hard_target_model_updates(self.target, self.network)
		elif self.algorithm == 'basic':
			self.target = self.network # target = network (no fixing case)
		elif self.coin_flip:
			self.target = target_network
		else:
			raise Exception("This should not happen.  Check boolean instance variables.")

	def calc_q_values(self, model, state):
		action_mask = np.ones([1, self.num_actions])
		q_values = model.predict_on_batch([state, action_mask])
		return q_values.flatten()

	def create_buffer(self, env):
		self.reset(env)

		# random sample of SARS pairs to prefill buffer
		for number in range(self.num_burn_in):
			S = self.preprocessor.get_state()
			A = np.random.randint(self.num_actions)
			s_prime, R, is_terminal, debug_info = env.step(A)
			self.preprocessor.add_state(s_prime)
			# get new processed state frames
			S_prime = self.preprocessor.get_state()
			R = self.preprocessor.process_reward(R)
			self.buffer.append(S, A, R, S_prime, is_terminal)
			if is_terminal:
				self.reset(env)

	# resets both environment and preprocessor			
	def reset(self, env):
		self.preprocessor.reset()
		self.preprocessor.add_state(env.reset())

	def get_minibatch(self):
		# get new processed state frames
		sample = self.buffer.sample(self.batch_size)

		# compute TD target
		true_output_masked = np.zeros([self.batch_size, self.num_actions])
		q_value_index = np.zeros([self.batch_size, self.num_actions])
		state = None
		for i in range(self.batch_size):
			true_output = sample[i].reward
			if not sample[i].is_terminal:
				q_values = self.calc_q_values(self.target, sample[i].next_state)

				if self.algorithm == 'double': #update network chooses actions
					q_values_update = self.calc_q_values(self.network, sample[i].next_state)
					A_prime = np.argmax(q_values_update)
				else:
					A_prime = np.argmax(q_values)

				true_output += self.gamma * q_values[A_prime]
			
			# output mask for appropriate action is one-hot vector
			true_output_masked[i][sample[i].action] = true_output

			# input mask for appropriate action is one-hot vector
			# must null out all other actions
			q_value_index[i][sample[i].action] = 1
			if state is None:
				state = sample[i].state
			else:
				state = np.append(state, sample[i].state, axis=0)

		return state, q_value_index, true_output_masked

	def step(self, env, A):
		# take action A
		s_prime, R, is_terminal, debug_info = env.step(A)

		# clip rewards from -1 to 1
		R = self.preprocessor.process_reward(R)

		# add state frame to frames
		self.preprocessor.add_state(s_prime)

		return R, is_terminal

	def select_action(self, S):
		# returns q_values and chosen action (network chooses action and evaluates)
		q_values = self.calc_q_values(self.network, S)
		q_selectors = q_values

		if self.coin_flip:
			q_selectors += self.calc_q_values(self.target, S)

		A = self.policy.select_action(q_selectors)
		return A, q_values

	def update_model(self, num_iters):
		minibatch, q_value_index, true_output_masked = self.get_minibatch()

		loss = self.network.train_on_batch([minibatch, q_value_index], true_output_masked)
		
		render("Loss on mini-batch [huber, mae] at " + str(num_iters) + " is " + str(loss), self.verbose)

	def switch_roles(self):
		should_switch = np.random.rand() < 0.5
		if should_switch:
			tmp = self.target
			self.target = self.network
			self.network = tmp

	def fit(self, env, num_iterations, eval_num, max_episode_length=None):
		"""Fit your model to the provided environment.
		Parameters
		----------
		env: gym.Env
			This is your Atari environment.
		num_iterations: int
			How many samples/updates to perform.
		max_episode_length: int
			Episode length before agent resets.
		"""
		self.create_buffer(env)
		
		# keep track of best weights, as well as highest reward
		best_reward = -float('inf')
		best_weights = None

		num_iters = 0
		while num_iters < num_iterations:

			self.reset(env)
			is_terminal = False

			steps = 0

			# start an episode
			while steps < max_episode_length and not is_terminal:
				# evaluate and save best_reward and best_weights
				if (num_iters % self.eval_freq == 0):
					avg_reward, avg_q, avg_steps, max_reward, std_dev_rewards = self.evaluate(env, eval_num)

					weights = self.network.get_weights()

					print(str(num_iters) + ': ' + str(avg_reward) + ', ' + str(avg_q) + ', ' + str(avg_steps) + ', ' + str(max_reward) + ', ' + str(std_dev_rewards))

					if avg_reward > best_reward:
						best_reward = avg_reward
						best_weights = weights

				# compute step and gather SARS pair
				S = self.preprocessor.get_state()
				A, q_values = self.select_action(S)
				R, is_terminal = self.step(env, A)
				S_prime = self.preprocessor.get_state()

				self.buffer.append(S, A, R, S_prime, is_terminal)

				num_iters += 1
				steps += 1
					
				if self.target_fixing and num_iters % self.target_update_freq == 0:
					get_hard_target_model_updates(self.target, self.network)

				# train on minibatch
				if num_iters % self.update_freq == 0:
					self.update_model(num_iters)
					if self.coin_flip:
						self.switch_roles()

		# record last 100_rewards
		avg_reward, avg_q, std_dev_rewards = self.evaluate(env, 100)
		print(str(num_iters) + '(final):' + str(avg_reward) + ',' + str(avg_q) + ',' + str(std_dev_rewards))

		# TODO SAVE BEST_WEIGHTS = best_weights

		return best_reward, best_weights


	def evaluate(self, env, num_episodes=20):
		"""Test your agent with a provided environment.

		You shouldn't update your network parameters here. Also if you
		have any layers that vary in behavior between train/test time
		(such as dropout or batch norm), you should set them to test.

		Basically run your policy on the environment and collect stats
		like cumulative reward, average episode length, etc.

		You can also call the render function here if you want to
		visually inspect your policy.
		"""
		total_reward = 0.0
		average_q_value = 0.0

		rewards = []

		# evaluation always uses greedy policy
		greedy_policy = GreedyEpsilonPolicy(0.05)

		total_steps = 0

		for i in range(num_episodes):
			reward = 0.0
			df = 1.0

			s = env.reset()
			self.preprocessor.reset()
			self.preprocessor.add_state(s)

			steps = 0
			max_q_val_sum = 0

			is_terminal = False

			while not is_terminal:
				S = self.preprocessor.get_state()

				steps += 1
				total_steps += 1

				q_values = self.calc_q_values(self.network, S)
				A = greedy_policy.select_action(q_values)

				max_q_val_sum += np.max(q_values)

				s_prime, R, is_terminal, debug_info = env.step(A)

				reward += R * df

				self.preprocessor.add_state(s_prime)

				df *= self.gamma
			
			total_reward += reward
			rewards.append(reward)
			average_q_value += max_q_val_sum / steps

		avg_q, avg_reward = average_q_value / num_episodes, total_reward / num_episodes

		avg_steps = total_steps / num_episodes

		return avg_reward, avg_q, avg_steps, np.max(rewards), np.std(rewards)

	def evaluate_random(self, env, num_episodes=20):
		def eval(env):
			total_reward = 0
			rewards = []

			# evaluation always uses greedy policy
			p = UniformRandomPolicy(self.num_actions)

			total_steps = 0
			for i in range(num_episodes):
				reward = 0
				s = env.reset()
				steps = 0
				is_terminal = False
				while not is_terminal:
					steps += 1
					total_steps += 1
					_, R, is_terminal, _ = env.step(p.select_action())
					reward += R			
				total_reward += reward
				rewards.append(reward)

			avg_reward = total_reward / num_episodes
			avg_steps = total_steps / num_episodes
			return avg_reward, avg_steps, np.max(rewards), np.std(rewards)

		for i in range(20):
			avg_reward, avg_steps, max_reward, std_dev_rewards = eval(env)
			print(str(avg_reward) + ',' + str(max_reward) + ',' + str(avg_steps) + ',' + str(std_dev_rewards))
