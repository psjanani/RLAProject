from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import gym


from modules.policy import GreedyEpsilonPolicy
import gym
import time

from keras.models import model_from_json
from modules.dqn_agent import DQNAgent
import numpy as np
from modules.preprocessors import HistoryPreprocessor
import argparse
import os

def run_random_policy(env, pred_model, args):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    gamma = 0.99
    preprocessor = HistoryPreprocessor((args.size, args.size), args.network_name,
                                       args.number_pred, args.coop, args.history)
    initial_state = env.reset()
    env.render()
    total_reward = 0.0
    average_q_values = [0.0] * args.number_pred
    rewards = []

    # evaluation always uses greedy policy
    greedy_policy = GreedyEpsilonPolicy(0.0)
    total_steps = 0
    max_episode_length = 1000
    for i in range(args.num_episodes):
        reward = 0.0
        df = 1.0

        env.random_start = False
        s = env.reset()
        preprocessor.reset()
        preprocessor.add_state(s)

        steps = 0
        max_q_val_sum = [0] * args.number_pred
        is_terminal = False

        while not is_terminal and steps < max_episode_length:
            S = preprocessor.get_state()

            steps += 1
            total_steps += 1
            A = {}
            action_string = ""

            for j in range(args.number_pred):
                model = pred_model[j]

                q_values = model.calc_q_values(model.network, S[j])

                A[j] = greedy_policy.select_action(q_values)
                action_string += str(A[j])
                max_q_val_sum[j] += np.max(q_values)

            s_prime, R, is_terminal, debug_info = env.step(action_string)

            env.render()


            R = preprocessor.process_reward(R)
            reward += R[0] * df  # same for each predator bc/ it's cooperative
            preprocessor.add_state(s_prime)
            df *= gamma

        total_reward += reward
        rewards.append(reward)
        for i in range(args.number_pred):
            average_q_values[i] += max_q_val_sum[i] / steps

    avg_q, avg_reward = np.sum(np.array(average_q_values)) / (args.num_episodes *
                                                              args.number_pred), total_reward / args.num_episodes
    avg_steps = total_steps / args.num_episodes
    return avg_reward, avg_q, avg_steps, np.max(rewards), np.std(rewards)



def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))

def calc_q_values(model, state, num_actions):
    action_mask = np.ones([1, num_actions])
    q_values = model.predict_on_batch([state, action_mask])
    return q_values.flatten()


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Pacman!')
    parser.add_argument('--algorithm', default='replay_target', help='One of basic, replay_target, double')
    parser.add_argument('--compet', default=False, type=bool, help='Coop or compete.')
    parser.add_argument('--debug_mode', default=False, type=bool, help='Whether or not to save states as images.')
    parser.add_argument('--env', default='PacmanEnvSmartPrey-v0', help='Env name')
    parser.add_argument('--num_episodes', default=25, type=int, help='Number of episodes to evaluate on.')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor (0, 1)')
    parser.add_argument('--history', default=1, type=int, help='number of frames that make up a state')
    parser.add_argument('--max_episode_length', default=500, type=int,
                        help='Max episode length (for training, not eval).')
    parser.add_argument('--network_name', default='deep',
                        help='Model Name: deep, stanford, linear, dueling, dueling_av, or dueling_max')
    parser.add_argument('--weight_path', default='/Users/janani/weights/single_agent', type=str,
                        help='To save weight at eval frequency')
    args = parser.parse_args()

    args.coop = not bool(args.compet)

    # create the environment
    env = gym.make('SpaceInvaders-v0')
    args.num_pred = env.num_agents / 2
    args.size = env.grid_size

    # uncomment next line to try the deterministic version
    # env = gym.make('Deterministic-4x4-FrozenLake-v0')

#    print_env_info(env)
    pred_model = {}
    mypath = args.weight_path + "/" + args.v
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    for i in range(args.number_pred):
        json_file = open('/Users/janani/deep_double.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/Users/janani/1400000-deep-double-252.5.hd5")
        pred_model[i] = loaded_model
    print("Loaded model from disk")
    for i in range(100):
        total_reward, num_steps = run_random_policy(env, pred_model, args)
        print (total_reward)
    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()