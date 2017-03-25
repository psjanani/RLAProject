from dqn_agent import *
from models import *
import gym
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from objectives import mean_huber_loss, huber_loss

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
        self.gamma = args.gamma
        self.algorithm = args.algorithm
        self.optimizer = optimizer
        self.loss = loss
        self.model_name = model_name
        if (model_name == 'linear'):
            self.m = LinearModel(args.channels, (args.dim, args.dim), args.num_actions)

        if (model_name == 'stanford'):
            self.m = StanfordModel(args.channels, (args.dim, args.dim), args.num_actions)

        if (model_name == 'deep' or 'dueling' in model_name):
            self.m = DeepQModel(args.channels, (args.dim, args.dim), args.num_actions, model_name)

    def create_model(self, env, args):
        self.model_init(args)
        self.args = args
        self.env = env

        for i in range(self.number_pred):
            model = self.m.create_model()
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mae'])
            buffer = None
            if (args.num_burn_in != 0):
                buffer = NaiveReplay(args.memory, not self.algorithm == 'basic', None)
            self.pred_model[i] = DQNAgent(i, model, buffer, self.preprocessor, None, args)

    def model_init(self, args):
        self.preprocessor = HistoryPreprocessor((args.dim, args.dim), args.network_name, args.channels, args.history)


    def fit(self, num_iterations, eval_num, max_episode_length=None):
        best_reward = -float('inf')
        best_weights = None

        for i in range(self.number_pred):
            self.pred_model[i].create_buffer(self.env)
        print self.number_pred
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
                    A[i], q_values[i] = self.pred_model[i].select_action(S)
                    action_string += str(A[i])

                R, is_terminal = self.step(action_string)
                S_prime = self.preprocessor.get_state()

                for i in range(self.number_pred):
                    model = self.pred_model[i]
                    model.buffer.append(S, A[i], R, S_prime, is_terminal)
                    if model.target_fixing and num_iters % model.target_update_freq == 0:
                        get_hard_target_model_updates(model.target, model.network)
                    if num_iters % model.update_freq == 0:
                        model.update_model(num_iters)
                        if model.coin_flip:
                            model.switch_roles()

                num_iters += 1
                steps += 1

        # record last 100_rewards
        avg_reward, avg_q, std_dev_rewards = self.evaluate(100)
        print(str(num_iters) + '(final):' + str(avg_reward) + ',' + str(avg_q) + ',' + str(std_dev_rewards))

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

    def evaluate(self, num_episodes=20):
        total_reward = 0.0
        average_q_value = 0.0
        rewards = []

        # evaluation always uses greedy policy
        greedy_policy = GreedyEpsilonPolicy(0.05)
        total_steps = 0

        for i in range(num_episodes):
            reward = 0.0
            df = 1.0

            s = self.env.reset()
            self.preprocessor.reset()
            self.preprocessor.add_state(s)

            steps = 0
            max_q_val_sum = [0] * self.number_pred
            is_terminal = False

            while not is_terminal:
                S = self.preprocessor.get_state()

                steps += 1
                total_steps += 1
                A = {}
                action_string = ""

                for i in range(self.number_pred):
                    model = self.pred_model[i]
                    q_values = model.calc_q_values(model.network, S)
                    A[i] = greedy_policy.select_action(q_values)
                    action_string += str(A[i])
                    max_q_val_sum[i] += np.max(q_values)

                s_prime, R, is_terminal, debug_info = self.env.step(A)
                reward += R * df
                self.preprocessor.add_state(s_prime)
                df *= self.gamma

            total_reward += reward
            rewards.append(reward)
            for i in range(self.number_pred):
                average_q_value[i] += max_q_val_sum[i] / steps

        avg_q, avg_reward = average_q_value / num_episodes, total_reward / num_episodes
        avg_steps = total_steps / num_episodes
        return avg_reward, avg_q, avg_steps, np.max(rewards), np.std(rewards)


