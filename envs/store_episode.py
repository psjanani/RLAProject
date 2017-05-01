from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import gym


from modules.policy import GreedyEpsilonPolicy
import gym
import time
import envs.pacman_envs
from keras.models import model_from_json
from modules.dqn_agent import DQNAgent
import numpy as np
from modules.preprocessors import HistoryPreprocessor
import argparse
import os
from os.path import expanduser

TIE_BREAK = 'max'
NO_NASH_CHOICE = 'best_sum'
FULL_INFO = True

def select_actions(q_values1, q_values2, epsilon):
    if FULL_INFO:
        A = select_joint_actions(q_values1, q_values2, epsilon)
        for i in range(len(A)):
            A[i] = str(A[i])
    else:
        A1 = select_joint_actions(q_values1, q_values1, epsilon)
        A2 = select_joint_actions(q_values2, q_values2, epsilon)
        A = [ str(A1[0]) , str(A2[0]) ]

    return A

def select_joint_actions(q_values1, q_values2, epsilon=None):
    threshold = np.random.rand()

    epsilon = 1.0 if epsilon is None else epsilon
    is_random = epsilon > threshold

    if is_random:
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
        if TIE_BREAK == 'max':
            nash_idx = np.argmax(nash_eq_qsum)
        elif TIE_BREAK == 'greedy':
            action1 = nash_eq[np.argmax(nash_eq_q1)][0]
            action2 = nash_eq[np.argmax(nash_eq_q2)][1]
            return [action1, action2]
        else:
            raise Exception("Unrecognized...")

        return nash_eq[nash_idx]
    else:
        if NO_NASH_CHOICE == 'best_sum':
            action1 = np.argmax(np.sum(payoffs1, axis=0))
            action2 = np.argmax(np.sum(payoffs2, axis=0))
        elif NO_NASH_CHOICE == 'best_max':
            action1 = np.argmax(np.max(payoffs1, axis=0))
            action2 = np.argmax(np.max(payoffs2, axis=0))
        elif NO_NASH_CHOICE == 'rando':
            action1 = np.random.randint(4)
            action2 = np.random.randint(4)
        else:
            raise Exception("Unrecognized...")

        return [action1, action2]

def calc_q_values(model, state, num_actions, expand_dims=False):
    if expand_dims:
        state = np.expand_dims(state, axis=0)

    action_mask = np.ones([1, num_actions])
    q_values = model.predict_on_batch([state, action_mask])
    return q_values.flatten()

def run_random_policy(env, model, args):
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
    preprocessor = HistoryPreprocessor((args.size, args.size), args.network_name,
                                       args.num_agents, args.coop, 'Amazon' in args.env, False, args.history)

    initial_state = env.reset()
    my_range = 1 if args.control == 'set_controller' else args.num_agents

    env.render()

    states = []
    actions = []

    # evaluation always uses greedy policy
    greedy_policy = GreedyEpsilonPolicy(args.epsilon)

    for i in range(1):
        env.random_start = False
        s = env.reset()
        preprocessor.reset()
        preprocessor.add_state(s)
        states.add(s)

        steps = 0
        is_terminal = False
        while not is_terminal and steps < args.max_test_episode_length:
            S = preprocessor.get_state()

            steps += 1

            if args.control == 'random':
                action_string = str(np.random.randint(4)) + str(np.random.randint(4))
            elif not args.control == 'nashq' and not args.controller_plus_nash:
                action_string = ""
                for j in range(my_range):
                    q_values = calc_q_values(model[j], S[j], args.num_actions)

                    A = greedy_policy.select_action(q_values)
                    if args.control == 'set_controller':
                        action_string += str(A / 4)
                        action_string += str(A % 4)
                    else: # IDQN or friend algorithm assume greedy control
                        if args.control == 'friend':
                            A /= 4
                        action_string += str(A)
            else:
                q_values = []
                q_values.append(calc_q_values(model[0], S[0], args.num_actions))

                if args.controller_plus_nash:
                    next_state_other = np.zeros([1, 8])
                    next_state_other[0, 0:3] = S[0][0, 3:6]
                    next_state_other[0, 3:6] = S[0][0, 0:3]
                    q_values.append(calc_q_values(model[0], next_state_other, args.num_actions))
                else:
                    q_values.append(calc_q_values(model[1], S[1], args.num_actions))
                action_string= select_actions(q_values[0], q_values[1], args.epsilon)

            s_prime, R, is_terminal, debug_info = env.step(action_string)

            states.append(s_prime)
            actions.append(action_string)
            preprocessor.add_state(s_prime)

    return states, actions

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Amazon!')
    parser.add_argument('--algorithm', default='replay_target', help='One of basic, replay_target, double')
    parser.add_argument('--compet', default=False, type=bool, help='Coop or compete.')
    parser.add_argument('--debug_mode', default=False, type=bool, help='Whether or not to save states as images.')
    parser.add_argument('--env', default='Amazon-FixedStart-v1', help='Env name')
    parser.add_argument('--gamma', default=0.95, type=float, help='discount factor (0, 1)')
    parser.add_argument('--history', default=1, type=int, help='number of frames that make up a state')
    parser.add_argument('--network_name', default='linear',
                        help='Model Name: deep, stanford, linear, dueling, dueling_av, or dueling_max')
    parser.add_argument('--weight_path', default='~/weights/', type=str,
                        help='To load weights')
    parser.add_argument('--v', required=True, type=str, help='experiment names, used for loading weights')
    parser.add_argument('--iter', required=True, type=int, help='the weights to load')
    parser.add_argument('--control', required=True, help='iqdn, nashq, friend, set_controller')
    parser.add_argument('--epsilon', type=float, default=0.00, help='Epsilon for greedy epsilon')
    parser.add_argument('--max_test_episode_length', type=int, default=10000)

    parser.add_argument('--controller_plus_nash', default=False, type=bool)

    EXPERIENCE_DIR = 'experience/'

    args = parser.parse_args()

    args.coop = not bool(args.compet)

    # create the environment
    env = gym.make(args.env)
    args.num_agents = env.num_agents
    args.size = env.grid_size

    if 'Pacman' in args.env:
        args.num_actions = 4 ** args.num_pred
        args.num_pred = env.num_predators
    elif 'Warehouse' in args.env:
        args.num_actions = 6
    elif 'Amazon' in args.env:
        exp = 1 if args.control == 'idqn' else args.num_agents
        args.num_actions = 4 ** exp
    else:
        args.num_actions = 4 ** args.num_agents

    my_range = 1 if args.control == 'set_controller' else args.num_agents

    pred_model = {}

    if not args.control == 'random':
        args.weight_path = expanduser(args.weight_path)
        mypath = args.weight_path + "/" + args.v
        if not os.path.isdir(mypath):
            os.makedirs(mypath)
        for i in range(my_range):
            json_file = open(args.weight_path + args.v +"/model" + str(i) + ".json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(args.weight_path + args.v + "/" + str(args.iter) + "_" + str(i) + ".hd5")
            pred_model[i] = loaded_model
        print("Loaded model from disk")
    
    actions, states = run_policy(env, pred_model, args)

    np.save(EXPERIENCE_DIR + args.v + '_actions.npy', np.array(actions))
    np.save(EXPERIENCE_DIR + args.v + '_actions.npy', np.array(states))

if __name__ == '__main__':
    main()