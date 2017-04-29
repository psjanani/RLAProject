from keras.optimizers import Adam
import numpy as np
from replay import NaiveReplay, Prioritized_Replay
from keras.models import Model
from utils import get_hard_target_model_updates, render
from modules.preprocessors import HistoryPreprocessor
from modules.policy import LinearDecayGreedyEpsilonPolicy, GreedyPolicy, \
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
	## Currently the buffer, preprocessor, env can be shared between multiple DQNS.
	def __init__(self, id, network, buffer, preprocessor, target_network, args):
		self.id = id
		self.network = network
		self.gamma = args.gamma
		self.network_name = args.network_name
		self.num_pred = args.num_agents

		if 'Pacman' in args.env:
			self.num_pred /= 2

		self.memory = args.memory
		self.num_actions = int(args.num_actions)
		self.target_update_freq = args.target_update_freq
		self.activation = args.activation
		self.num_burn_in = int(args.num_burn_in)
		self.batch_size = args.batch_size
		self.smart_burn_in = bool(args.smart_burn_in)
		self.algorithm = args.algorithm
		self.set_controller = args.set_controller
		self.update_freq = int(args.update_freq)
		# 'basic' algorithm has no target fixing or experience replay
		self.target_fixing = not self.algorithm == 'basic'
		self.verbose = args.verbose > 0
		self.preprocessor = preprocessor
		self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.end_epsilon, args.num_decay_steps)
		self.buffer = buffer

		if self.target_fixing:
			self.target = Model.from_config(self.network.get_config())
			get_hard_target_model_updates(self.target, self.network)
		elif self.algorithm == 'basic':
			self.target = self.network # target = network (no fixing case)
		else:
			raise Exception("This should not happen.  Check boolean instance variables.")

	def save(self, path):
		self.network.save(path)

	def calc_q_values(self, model, state):
		if len(state.shape) < 2:
			state = np.expand_dims(state, axis=0)

		action_mask = np.ones([1, self.num_actions])
		q_values = model.predict_on_batch([state, action_mask])

		return q_values.flatten()

	def create_buffer(self, env):
		self.reset(env)

		S = self.preprocessor.get_state(self.id)

		if self.smart_burn_in:
			env.quick_burn_in()


		# random sample of SARS pairs to prefill buffer
		for number in range(self.num_burn_in):
			action_str = [0] * self.num_pred
			A = np.random.randint(self.num_actions)

			other_action = int(A % 4)
			my_action = int(A // 4)

			action_str[0] = str(my_action)
			action_str[1] = str(other_action)

			s_prime, R, is_terminal, debug_info = env.step("".join(action_str))

			# retrieve state (from env) since it's chosen by env
			if self.smart_burn_in:
				A = env.latest_first_pred_action

			self.preprocessor.add_state(s_prime)

			# get new processed state frames
			S_prime = self.preprocessor.get_state(self.id)
			R = self.preprocessor.process_reward(R)
			self.buffer.append(S, A, R[self.id], S_prime, is_terminal)

			S = S_prime

			if is_terminal:
				self.reset(env)

	# resets both environment and preprocessor			
	def reset(self, env):
		self.preprocessor.reset()
		self.preprocessor.add_state(env.reset())

	def get_minibatch(self):
		# get new processed state frames
		if (self.algorithm == 'replay_target' or self.algorithm == 'double'):
		    sample = self.buffer.sample(self.batch_size)
		else:
			sample, w, id1 = self.buffer.sample(self.batch_size)
		true_output_masked = np.zeros([self.batch_size, self.num_actions])
		q_value_index = np.zeros([self.batch_size, self.num_actions])
		# compute TD target
		if self.set_controller:
			true_output_masked = np.zeros([self.batch_size, self.num_actions])
			q_value_index = np.zeros([self.batch_size, self.num_actions])

		state = None
		delta = []
		for i in range(self.batch_size):
			true_output = sample[i].reward

			S = sample[i].state
			S_prime = sample[i].next_state

			if not sample[i].is_terminal:
				q_values = self.calc_q_values(self.target, sample[i].next_state)

				if self.algorithm == 'double': #update network chooses actions
					q_values_update = self.calc_q_values(self.network, sample[i].next_state)
					A_prime = np.argmax(q_values_update)
				else:
					A_prime = np.argmax(q_values)

					true_output += self.gamma * q_values[A_prime]
			delta.append(true_output - self.calc_q_values(self.target, sample[i].state)[sample[i].action])
			# output mask for appropriate action is one-hot vector
			true_output_masked[i][sample[i].action] = true_output

			# input mask for appropriate action is one-hot vector
			# must null out all other actions
			q_value_index[i][sample[i].action] = 1
			if state is None:
				state = S
			else:
				state = np.append(state, S, axis=0)
		if (self.algorithm == 'priority'):
			self.buffer.update_priority(delta, id1)

		return state, q_value_index, true_output_masked

	def select_action(self, S, expand_dims=False):
		# returns q_values and chosen action (network chooses action and evaluates)
		q_values = self.calc_q_values(self.network, S)
		q_selectors = q_values
		A = self.policy.select_action(q_selectors)
		return A, q_values

	def update_model(self, num_iters):
		minibatch, q_value_index, true_output_masked = self.get_minibatch()
		loss = self.network.train_on_batch([minibatch, q_value_index], true_output_masked)
	# render("Loss on mini-batch [huber, mae] at " + str(num_iters) + " is " + str(loss), self.verbose)

	def switch_roles(self):
		should_switch = np.random.rand() < 0.5
		if should_switch:
			tmp = self.target
			self.target = self.network
			self.network = tmp
