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
        self.number_pred = number_agents 

        if 'Pacman' in args.env:
            self.number_pred /= 2

        self.pred_model = {}
        self.coop = args.coop
        self.gamma = args.gamma
        self.debug_mode = args.debug_mode
        self.max_test_episode_length = args.max_test_episode_length
        self.agent_dissemination_freq = args.agent_dissemination_freq
        self.algorithm = args.algorithm
        self.optimizer = optimizer
        self.eval_freq = args.eval_freq
        self.eval_num = args.eval_num

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
        self.preprocessor = HistoryPreprocessor((args.dim, args.dim), args.network_name, self.number_pred, self.coop, 'Amazon' in args.env, args.history)

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
                A = {}
                q_values = {}
                action_string = ""
                for i in range(self.number_pred):
                    A[i], q_values[i] = self.pred_model[i].select_action(S[i])
                    action_string += str(A[i])

                R, is_terminal = self.step(action_string)

                if(sum(R) != 0):
                    print(R)

                S_prime = self.preprocessor.get_state()

                if num_iters % self.eval_freq == 0:

                    avg_reward, avg_q, avg_steps, max_reward, std_dev_rewards = self.evaluate(self.eval_num, self.max_test_episode_length, num_iters % (self.eval_freq * 5) == 0)# num_iters % 50000 == 0)

                    print(str(num_iters) + ':\tavg_reward=' + str(avg_reward) + '\tavg_q=' + str(avg_q) + '\tavg_steps=' \
                        + str(avg_steps) + '\tmax_reward=' + str(max_reward) + '\tstd_dev_reward=' + str(std_dev_rewards))
                    if self.args.save_weights and num_iters % (5 * self.eval_freq) == 0:
                        for i in range(self.number_pred):
                            model = self.pred_model[i].network
                            model.save(self.args.weight_path + self.args.v + "/" + str(num_iters) + "_" + str(i) + ".hd5")

                for i in range(self.number_pred):
                    model = self.pred_model[i]
                    if i > 0 and self.args.solo_train:
                        self.pred_model[0].buffer.append(S[i], A[i], R[i], S_prime[i], is_terminal)

                        if num_iters % self.agent_dissemination_freq == 0:
                            get_hard_target_model_updates(self.pred_model[0].network, model.network)
                    else:
                        model.buffer.append(S[i], A[i], R[i], S_prime[i], is_terminal)
                        if model.target_fixing and num_iters % model.target_update_freq == 0:
                            get_hard_target_model_updates(model.target, model.network)
                        if num_iters % model.update_freq == 0 or is_terminal:
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
        average_q_values = [0.0] * self.number_pred
        rewards = []

        # evaluation always uses greedy policy
        greedy_policy = GreedyEpsilonPolicy(0.1)
        total_steps = 0

        for i in range(num_episodes):
            reward = 0.0
            df = 1.0

            self.env.random_start = False
            s = self.env.reset()
            self.preprocessor.reset()
            self.preprocessor.add_state(s)

            steps = 0
            max_q_val_sum = [0] * self.number_pred
            is_terminal = False

            while not is_terminal and steps < max_episode_length:
                S = self.preprocessor.get_state()

                steps += 1
                total_steps += 1
                A = {}
                action_string = ""

                for j in range(self.number_pred):
                    model = self.pred_model[j]

                    q_values = model.calc_q_values(model.network, S[j])

                    A[j] = greedy_policy.select_action(q_values)
                    action_string += str(A[j])
                    max_q_val_sum[j] += np.max(q_values)

                s_prime, R, is_terminal, debug_info = self.env.step(action_string)

                if to_render and i == 0:
                    self.env.render()
                    print('\n')
                    print q_values
                    print('\n')

                    if len(R) > 0:
                        print(R)
                        print('\n')

                if self.debug_mode:
                    save_states_as_images(S)

                R = self.preprocessor.process_reward(R)
                reward += R[0] * df # same for each predator bc/ it's cooperative
                self.preprocessor.add_state(s_prime)
                df *= self.gamma

            total_reward += reward
            rewards.append(reward)
            for i in range(self.number_pred):
                average_q_values[i] += max_q_val_sum[i] / steps

        avg_q, avg_reward = np.sum(np.array(average_q_values)) / (num_episodes * self.number_pred), total_reward / num_episodes
        avg_steps = total_steps / num_episodes
        return avg_reward, avg_q, avg_steps, np.max(rewards), np.std(rewards)

