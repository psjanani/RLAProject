#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym

from modules.dqn_agent import DQNAgent
from modules.objectives import mean_huber_loss, huber_loss
from modules.utils import argrender
from modules.models import *
from modules.multi_agent import *
import envs.pacman_envs

def make_assertions(args):
    algo = args.algorithm
    network_name = args.network_name
    assert network_name == 'stanford' or network_name == 'linear' or network_name == 'deep' or 'dueling' in network_name
    assert algo == 'basic' or algo == 'replay_target' or algo == 'double'

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Pacman!')
    parser.add_argument('--algorithm', default='basic', help='One of basic, replay_target, double')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--compet', default=False, type=bool, help='Coop or compete.')
    parser.add_argument('--debug_mode', default=False, type=bool, help='Whether or not to save states as images.')
    parser.add_argument('--end_epsilon', default=0.05, type=float, help='Steady state epsilon')
    parser.add_argument('--env', default='PacmanEnv-v0', help='Env name')
    parser.add_argument('--eval_freq', default=1e4, type=int, help='Number frames in between evaluations')
    parser.add_argument('--eval_num', default=25, type=int, help='Number of episodes to evaluate on.')
    parser.add_argument('--eval_random', default=False, type=bool, help='To render eval on random policy or not.')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor (0, 1)')
    parser.add_argument('--history', default=1, type=int, help='number of frames that make up a state')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial epsilon pre-decay')
    parser.add_argument('--loss', default='mean_huber', help='mean_huber, huber, mae, or mse.')
    parser.add_argument('--lr', default=0.00025, type=float, help='(initial) learning rate')
    parser.add_argument('--max_episode_length', default=500, type=int, help='Max episode length (for training, not eval).')
    parser.add_argument('--memory', default=1e6, type=int, help='size of buffer for experience replay')
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--solo_train', default=True, type=bool, help='Whether to train models one at a time or simultaneously.')
    parser.add_argument('--agent_dissemination_freq', default=1e4, type=int, help='If solo training, how frequently to copy trained weights to other untrained agents.')
    parser.add_argument('--network_name', default='deep', help='Model Name: deep, stanford, linear, dueling, dueling_av, or dueling_max')
    parser.add_argument('--optimizer', default='adam', help='one of sgd, rmsprop, and adam')
    parser.add_argument('--num_burn_in', default=5e4, type=int, help='Buffer size pre-training.')
    parser.add_argument('--num_decay_steps', default=1e6, type=int, help='Epsilon policy decay length')
    parser.add_argument('--num_iterations', default=1e6, type=int, help='Number frames visited for training.')
    parser.add_argument('--smart_burn_in', default=False, type=bool)
    parser.add_argument('--target_update_freq', default=1e4, type=int, help='Target Update frequency. Only applies to algorithm==replay_target, double, dueling.')
    parser.add_argument('--test_mode', default=False, type=bool, help='Just render evaluation.')
    parser.add_argument('--update_freq', default=1, type=int, help='Update frequency.')
    parser.add_argument('--verbose', default=2, type=int, help='0 - no output. 1 - loss and eval.  2 - loss, eval, and model summary.')
    parser.add_argument('--save_weights', default=True, type=bool, help='To save weight at eval frequency')
    parser.add_argument('--weight_path', default='/Users/griffinadams/Desktop/RLAProject/weights/007_double_agent', type=str, help='To save weight at eval frequency')

    args = parser.parse_args()
    
    args.coop = not bool(args.compet)

    make_assertions(args)

    if args.test_mode:
        args.verbose = 0

    # make environment
    env = gym.make(args.env)
    args.num_agents = env.num_agents

    args.dim = env.grid_size

    if 'Pacman' in args.env:
        args.num_actions = 4
    elif 'Warehouse' in args.env:
        args.num_actions = 6
    else:
        args.num_actions = env.action_space.n
        ## not sure

    if args.verbose == 2:
        argrender(args)

    if args.optimizer == 'rmsprop':
        optimizer = RMSprop(lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(lr=args.lr)
    elif args.optimizer == 'nadam':
        optimizer = Nadam(lr=args.lr)
    else:
        raise Exception("Only rmsprop, sgd, nadam, and adam currently supported.")

    if args.loss == 'mean_huber':
        loss = mean_huber_loss
    elif args.loss == 'huber':
        loss = huber_loss
    else:
        loss = args.loss

    # Create the multi-Agent setting
    multiagent = IndependentDQN(args.num_agents, args.network_name, args, optimizer, loss)
    multiagent.create_model(env, args)
    if args.save_weights:
        for i in range(multiagent.number_pred):
            model = multiagent.pred_model[i].network
            model_json = model.to_json()
            with open(args.weight_path +"model" + str(i) + ".json", "w") as json_file:
                json_file.write(model_json)
    multiagent.fit(args.num_iterations, args.eval_num, args.max_episode_length)


if __name__ == '__main__':
    main()
