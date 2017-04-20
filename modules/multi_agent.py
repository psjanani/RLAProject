from dqn_agent import *
from models import *
import gym
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from objectives import mean_huber_loss, huber_loss
from utils import save_states_as_images

from time import sleep

class MultiAgent:
    # A template class for handling the multiple agents and their interactions with environment.
    def evaluate(self, num_episodes):
        pass
    # Training the model
    def fit(self, num_iterations, eval_num, max_episode_length=None):
        pass
    # Interacting with the environment
    def step(self, A):
        pass

class IndependentDQN(MultiAgent):
    # Simplest case where each agent has an autonomous DQN
    # The buffers are different for each DQN, history Preprocessor is the same. During the training,
    # the interaction with the environment is coordinated.
    def __init__(self, number_agents, model_name, args, optimizer, loss):
        self.number_pred = number_agents / 2
        self.pred_model = {}
        self.coop = args.coop

        self.tie_break = args.tie_break
        self.no_nash_choice = args.no_nash_choice

        self.gamma = args.gamma
        self.debug_mode = args.debug_mode
        self.full_info = args.full_info
        self.max_test_episode_length = args.max_test_episode_length
        self.agent_dissemination_freq = args.agent_dissemination_freq
        self.should_move_prob = args.should_move_prob
        self.algorithm = args.algorithm
        self.optimizer = optimizer
        self.eval_freq = args.eval_freq
        self.eval_num = args.eval_num
        self.initial_epsilon = args.initial_epsilon
        self.num_decay_steps = args.num_decay_steps
        self.end_epsilon = args.end_epsilon
        self.loss = loss
        self.model_name = model_name
        if (model_name == 'linear'):
            self.m = LinearModel((args.dim, args.dim), args.num_actions)

        if (model_name == 'stanford'):
            self.m = StanfordModel((args.dim, args.dim), args.num_actions)

        if (model_name == 'deep' or 'dueling' in model_name):
            self.m = DeepQModel((args.dim, args.dim), args.num_actions, model_name)

    def create_model(self, env, args):
        self.model_init(args)
        self.args = args
        self.env = env

        for i in range(self.number_pred):
            # if train one at a time every model after 1st is a copy of the first's weights
            buffer = None
            if i > 0 and self.args.solo_train:
                model = Model.from_config(self.pred_model[0].network.get_config())
                get_hard_target_model_updates(model, self.pred_model[0].network)
            else:
                model = self.m.create_model()
                model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mae'])
                if (args.num_burn_in != 0):
                    buffer = NaiveReplay(args.memory, True, None)
            self.pred_model[i] = DQNAgent(i, model, buffer, self.preprocessor, None, args)

        return model

    def model_init(self, args):
        self.preprocessor = HistoryPreprocessor((args.dim, args.dim), args.network_name, self.number_pred, self.coop, args.history)

    def select_actions(self, q_values1, q_values2, num_iters):
        if self.full_info:
            A = self.select_joint_actions(q_values1, q_values2, num_iters)
            for i in range(len(A)):
                A[i] = str(A[i])
        else:
            A1 = self.select_joint_actions(q_values1, q_values1, num_iters)
            A2 = self.select_joint_actions(q_values2, q_values2, num_iters)
            A = [ str(A1[0]) , str(A2[0]) ]

        return A

    def select_joint_actions(self, q_values1, q_values2, num_iters):
        threshold = np.random.rand()

        perc = min(num_iters / self.num_decay_steps, 1.0)
        epsilon = perc * self.end_epsilon + ( 1.0 - perc) * self.initial_epsilon

        is_random = epsilon > threshold

        if is_random:
            should_move = np.random.rand() < self.should_move_prob

            if should_move:
                movable_actions = self.env.movable_actions()

                movable_actions1 = movable_actions[0]
                num_movable_actions1 = len(movable_actions1)

                movable_actions2 = movable_actions[1]
                num_movable_actions2 = len(movable_actions2)
                
                action1_idx = np.random.randint(num_movable_actions1)
                action2_idx = np.random.randint(num_movable_actions2)

                action1 = movable_actions1[action1_idx]
                action2 = movable_actions2[action2_idx]
            else:
                action1 = np.random.randint(4)
                action2 = np.random.randint(4)
            return [action1, action2]

        payoffs1 = np.reshape(q_values1, [4, 4])
        payoffs2 = np.reshape(q_values2, [4, 4])

        br1 = np.argmax(payoffs1, axis=1)
        br2 = np.argmax(payoffs2, axis=1)

        nash_eq = []
        nash_eq_q1 = []
        nash_eq_q2 = []
        nash_eq_qsum = []

        for i in range(4):
            action1 = br1[i]
            action2 = i

            other_action2 = br2[action1]

            if action2 == other_action2:                
                nash_eq.append([str(action1), str(action2)])

                q_val1 = payoffs1[action1, action2]
                q_val2 = payoffs2[action2, action1]

                nash_eq_q1.append(q_val1)
                nash_eq_q2.append(q_val2)
                nash_eq_qsum.append(q_val1 + q_val2)

        if len(nash_eq) > 0:
            ## if we maxize joint
            if self.tie_break == 'max':
                nash_idx = np.argmax(nash_eq_qsum)
            elif self.tie_break == 'greedy':
                action1 = nash_eq[np.argmax(nash_eq_q1)][0]
                action2 = nash_eq[np.argmax(nash_eq_q2)][1]
                return [action1, action2]
            else:
                nash_idx = np.random.randint(len(nash_eq))

            return nash_eq[nash_idx]
        else:
            if self.no_nash_choice == 'best_sum':
                action1 = np.argmax(np.sum(payoffs1, axis=0))
                action2 = np.argmax(np.sum(payoffs2, axis=0))
            elif self.no_nash_choice == 'best_max':
                action1 = np.argmax(np.max(payoffs1, axis=0))
                action2 = np.argmax(np.max(payoffs2, axis=0))
            else: # must be random
                action1 = np.random.randint(4)
                action2 = np.random.randint(4)
                return [action1, action2]

            return [ action1, action2 ]


    def fit(self, num_iterations, eval_num, max_episode_length=None):
        best_reward = -float('inf')
        best_weights = None

        for i in range(self.number_pred):
            if not self.args.solo_train or i == 0:
                self.pred_model[i].create_buffer(self.env)

        num_iters = 0
        while num_iters < num_iterations:
            self.pred_model[0].reset(self.env)
            is_terminal = False

            steps = 0
            # start an episode
            while steps < max_episode_length and not is_terminal:
                # compute step and gather SARS pair
                S = self.preprocessor.get_state()
                q_values = []
                for i in range(self.number_pred):
                    q_values.append(self.pred_model[i].calc_q_values(self.pred_model[i].network, S[i]))

                A = self.select_actions(q_values[0], q_values[1], num_iters)

                R, is_terminal = self.step(A)
                S_prime = self.preprocessor.get_state()

                if num_iters % self.eval_freq == 0:
                    avg_reward, avg_steps, max_reward, std_dev_rewards = self.evaluate(self.eval_num, self.max_test_episode_length, num_iters % (self.eval_freq * 5) == 0)
                    print(str(num_iters) + ':\tavg_reward=' + str(avg_reward) + '\tavg_steps=' \
                        + str(avg_steps) + '\tmax_reward=' + str(max_reward) + '\tstd_dev_reward=' + str(std_dev_rewards))
                    if self.args.save_weights and num_iters % (5 * self.eval_freq) == 0:
                        for i in range(self.number_pred):
                            model = self.pred_model[i].network
                            model.save(self.args.weight_path + self.args.v + "/" + str(num_iters) + "_" + str(i) + ".hd5")

                for i in range(self.number_pred):
                    model = self.pred_model[i]
                    if i > 0 and self.args.solo_train:
                        my_A = int(A[i]) * 4 + int(A[0])

                        # print my_A

                        self.pred_model[0].buffer.append(S[i], my_A, R[i], S_prime[i], is_terminal)

                        if num_iters % self.agent_dissemination_freq == 0:
                            get_hard_target_model_updates(self.pred_model[0].network, model.network)
                    else:
                        other_idx = 0 if i == 1 else 0
                        my_A = int(A[i]) * 4 + int(A[other_idx])

                        model.buffer.append(S[i], my_A, R[i], S_prime[i], is_terminal)
                        if model.target_fixing and num_iters % model.target_update_freq == 0:
                            get_hard_target_model_updates(model.target, model.network)
                        if num_iters % model.update_freq == 0:
                            model.update_model(num_iters)
                            if model.coin_flip:
                                model.switch_roles()

                num_iters += 1
                steps += 1

                if num_iters > num_iterations:
                    break

        model.save('end_model.h5')

        # record last 100_rewards
        avg_reward, avg_q, avg_steps, max_reward, std_dev_rewards = self.evaluate(self.eval_num, self.max_test_episode_length, True)
        print(str(num_iters) + '(final):\tavg_reward=' + str(avg_reward) + '\tavg_q=' + str(avg_q) + '\tavg_steps=' \
            + str(avg_steps) + '\tmax_reward=' + str(max_reward) + '\tstd_dev_reward=' + str(std_dev_rewards))

        if self.args.save_weights:
            for i in range(self.number_pred):
                model = self.pred_model[i]
                model.save(self.args.weight_path+ self.args.v + "/"+ str(num_iters)+"_"+str(i) + ".hd5")
        # TODO SAVE BEST_WEIGHTS = best_weights
        if(type(best_reward)==float):
            print best_reward
        return best_reward, best_weights

    def step(self, A):
        # take action A
        s_prime, R, is_terminal, debug_info = self.env.step(A)
        # clip rewards from -1 to 1
        R = self.preprocessor.process_reward(R)
        # add state frame to frames
        self.preprocessor.add_state(s_prime)
        return R, is_terminal

    def evaluate(self, num_episodes, max_episode_length, to_render):
        total_reward = 0.0
        rewards = []

        # evaluation always uses greedy policy
        greedy_policy = GreedyEpsilonPolicy(0.0)
        total_steps = 0

        for ne in range(num_episodes):
            reward = 0.0
            df = 1.0

            self.env.random_start = False
            s = self.env.reset()
            self.preprocessor.reset()
            self.preprocessor.add_state(s)

            steps = 0
            is_terminal = False

            while not is_terminal and steps < max_episode_length:
                S = self.preprocessor.get_state()

                steps += 1
                total_steps += 1

                q_values = []
                for i in range(self.number_pred):
                    q_values.append(self.pred_model[i].calc_q_values(self.pred_model[i].network, S[i]))

                A = self.select_actions(q_values[0], q_values[1], 10000000)

                for i in range(len(A)):
                    A[i] = str(A[i])

                s_prime, R, is_terminal, debug_info = self.env.step(A)

                # if to_render and ne == 0:
                #     self.env.render()
                #     print('\n')

                R = self.preprocessor.process_reward(R)
                reward += R[0] * df # same for each predator bc/ it's cooperative
                self.preprocessor.add_state(s_prime)
                df *= self.gamma

            total_reward += reward
            rewards.append(reward)


        avg_reward = total_reward / num_episodes
        avg_steps = total_steps / num_episodes
        return avg_reward, avg_steps, np.max(rewards), np.std(rewards)

