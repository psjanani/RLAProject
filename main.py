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
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--algorithm', default='basic', help='One of basic, replay_target, double')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--end_epsilon', default=0.1, type=float, help='Steady state epsilon')
    parser.add_argument('--env', default='PacmanEnv-v0', help='Env name')
    parser.add_argument('--eval_freq', default=1e4, type=int, help='Number frames in between evaluations')
    parser.add_argument('--eval_num', default=20, type=int, help='Number of episodes to evaluate on.')
    parser.add_argument('--eval_random', default=False, type=bool, help='To render eval on random policy or not.')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor (0, 1)')
    parser.add_argument('--history', default=1, type=int, help='number of frames that make up a state')
    parser.add_argument('--num_agents', default=4, type=int, help='number of agents = (2* num_pred)')
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial epsilon pre-decay')
    parser.add_argument('--loss', default='mean_huber', help='mean_huber, huber, mae, or mse.')
    parser.add_argument('--lr', default=0.00025, type=float, help='(initial) learning rate')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='Max episode length (for training, not eval).')
    parser.add_argument('--memory', default=1e6, type=int, help='size of buffer for experience replay')
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--network_name', default='stanford', help='Model Name: deep, stanford, linear, dueling, dueling_av, or dueling_max')
    parser.add_argument('--optimizer', default='adam', help='one of sgd, rmsprop, and adam')
    parser.add_argument('--num_burn_in', default=5e4, type=int, help='Buffer size pre-training.')
    parser.add_argument('--num_decay_steps', default=1e6, type=int, help='Epsilon policy decay length')
    parser.add_argument('--num_iterations', default=5e7, type=int, help='Number frames visited for training.')
    parser.add_argument('--target_update_freq', default=1e4, type=int, help='Target Update frequency. Only applies to algorithm==replay_target, double, dueling.')
    parser.add_argument('--update_freq', default=1, type=int, help='Update frequency.')
    parser.add_argument('--verbose', default=2, type=int, help='0 - no output. 1 - loss and eval.  2 - loss, eval, and model summary.')

    args = parser.parse_args()

    make_assertions(args)

    # basic uses no experience replay so
    # buffer only holds latest #batch_size examples
    if args.algorithm == 'basic':
        args.memory = args.batch_size
        args.num_burn_in = args.batch_size

    # make environment
    env = gym.make(args.env)

    args.dim = env.grid_size

    if 'Pacman' in args.env:
        args.num_actions = 4
        args.channels = 3
    elif 'Warehouse' in args.env:
        args.num_actions = 6
        args.channels = 3
    else:
        args.num_actions = env.action_space.n
        ## not sure

    if args.verbose == 2:
        argrender(args)

    # Create the multi-Agent setting
    multiagent = IndependentDQN(args.num_agents, args.network_name, args.channels, (args.dim, args.dim), args.num_actions)
    multiagent.create_model(env, args)
    multiagent.fit(args.num_iterations, args.eval_num, args.max_episode_length)


if __name__ == '__main__':
    main()