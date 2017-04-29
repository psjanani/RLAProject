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
import envs.amazon_envs
from os.path import expanduser

def make_assertions(args):
    algo = args.algorithm
    network_name = args.network_name
    assert network_name == 'stanford' or network_name == 'linear' or 'deep' in network_name or 'dueling' in network_name
    assert algo == 'basic' or algo == 'replay_target' or algo == 'double' or algo == 'priority'

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Amazon :P')
    parser.add_argument('--algorithm', default='replay_target', help='One of basic, priority, replay_target, double')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--compet', default=False, type=bool, help='Coop or compete.')
    parser.add_argument('--debug_mode', default=False, type=bool, help='Whether or not to save states as images.')
    parser.add_argument('--decay', default=1e-6, type=float, help="Learning Rate decay")
    parser.add_argument('--end_epsilon', default=0.5, type=float, help='Steady state epsilon')

    parser.add_argument('--activation', default='sigmoid', help='Activation for linear model: sigmoid, tanh, or relu recommended.')

    parser.add_argument('--env', default='Amazon-v1', help='Env name')
    parser.add_argument('--eval_freq', default=10000, type=int, help='Number frames in between evaluations')
    parser.add_argument('--eval_num', default=10, type=int, help='Number of episodes to evaluate on.')
    parser.add_argument('--eval_random', default=False, type=bool, help='To render eval on random policy or not.')
    parser.add_argument('--gamma', default=0.95, type=float, help='discount factor (0, 1)')
    parser.add_argument('--history', default=1, type=int, help='number of frames that make up a state')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial epsilon pre-decay')
    parser.add_argument('--loss', default='mean_huber', help='mean_huber, huber, mae, or mse.')
    parser.add_argument('--lr', default=0.0001, type=float, help='(initial) learning rate')
    parser.add_argument('--max_test_episode_length', default=50, type=int, help='Max episode length for testing.')
    parser.add_argument('--max_episode_length', default=75, type=int, help='Max episode length (for training, not eval).')
    parser.add_argument('--memory', default=1e6, type=int, help='size of buffer for experience replay')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--solo_train', default=False, type=bool, help='Whether to train models one at a time or simultaneously.')

    parser.add_argument('--private', default=False, type=bool, help='Whether or not predators have access to each other\'s q-values')
    parser.add_argument('--should_move_prob', default=0.0, type=float)

    parser.add_argument('--agent_dissemination_freq', default=1e4, type=int, help='If solo training, how frequently to copy trained weights to other untrained agents.')
    parser.add_argument('--network_name', default='linear', help='Model Name: deep, stanford, linear, dueling, dueling_av, or dueling_max')
    parser.add_argument('--optimizer', default='adam', help='one of sgd, rmsprop, and adam')

    parser.add_argument('--num_burn_in', default=5e4, type=int, help='Buffer size pre-training.')
    parser.add_argument('--tie_break', default='max', help='how to break ties among nash equilibria.')
    parser.add_argument('--no_nash_choice', default='best_sum', help='how to choose when no nash equilibrium.  best_sum (action with best sum of q_values) or best_max (action with highest overall q.')
    
    parser.add_argument('--decay_factor', default=0.75, type=float, help='Percentage decay length over all iterations')
    parser.add_argument('--num_iterations', default=1e6, type=int, help='Number frames visited for training.')
    parser.add_argument('--smart_burn_in', default=False, type=bool)
    parser.add_argument('--set_controller', default=True, type=bool)
    parser.add_argument('--target_update_freq', default=1e4, type=int, help='Target Update frequency. Only applies to algorithm==replay_target, double, dueling.')
    parser.add_argument('--test_mode', default=False, type=bool, help='Just render evaluation.')
    parser.add_argument('--update_freq', default=2, type=int, help='Update frequency.')
    parser.add_argument('--verbose', default=2, type=int, help='0 - no output. 1 - loss and eval.  2 - loss, eval, and model summary.')
    parser.add_argument('--save_weights', default=True, type=bool, help='To save weight at eval frequency')
    parser.add_argument('--weight_path', default='~/weights/', type=str, help='To save weight at eval frequency')

    parser.add_argument('--v', default= 'def', type =str, help='experiment names, used for storing weights')


    parser.add_argument('--joint', default=False, type=bool, help='Whether to model single or joint action space') # 4 or 16
    parser.add_argument('--single_train', default=False, type=bool) # whether one agent (still state space of 8) xxx 000 xx , xxx yyy zz

    args = parser.parse_args()
    assert not (args.joint and args.single_train)

    args.num_iterations = args.update_freq * args.num_iterations

    args.num_decay_steps = int(args.num_iterations * args.decay_factor)
    args.coop = not bool(args.compet)
    args.full_info = not bool(args.private)

    if args.single_train:
        env_name = args.env.split('-')
        args.env = env_name[0] + '-Single-' + env_name[1]

    make_assertions(args)

    if args.test_mode:
        args.verbose = 0

    # make environment
    env = gym.make(args.env)
    env.set_single_train(args.single_train)
    args.num_agents = env.num_agents

    if args.v == 'def':
        print "You might want to name your experiment for later reference."

    args.dim = env.grid_size

    if 'Pacman' in args.env:
        args.num_actions = 4 ** args.num_pred
        args.num_pred = env.num_predators
    elif 'Warehouse' in args.env:
        args.num_actions = 6
    elif 'Amazon' in args.env:
        exp = args.num_agents if args.joint else 1
        args.num_actions = 4 ** exp
    else:
        args.num_actions = 4 ** args.num_agents

        ## not sure
    args.weight_path = expanduser(args.weight_path)
    mypath = args.weight_path + "/" + args.v
    if not os.path.isdir(mypath):
        os.makedirs(mypath)

    LOG_FILENAME = args.weight_path + args.v + ".log"
    f = open(LOG_FILENAME, 'w')
    for key in vars(env):
        f.write(str(key) + '=' + str(getattr(env, key)) + "\n")
    for arg in vars(args):
        f.write(arg + '=' + str(getattr(args, arg)) + "\n")
    f.close()

    if args.verbose == 2:
        argrender(args)

    if args.optimizer == 'rmsprop':
        optimizer = RMSprop()
    elif args.optimizer == 'sgd':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = Adam(lr=args.lr, decay=args.decay)
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
    multiagent = DIALAgent(args.num_agents, args.network_name, args, optimizer, loss)
    multiagent.create_model(env, args)
    if args.save_weights:
        myrange = 1 if args.single_train or args.set_controller else multiagent.number_pred
        for i in range(myrange):
            model = multiagent.pred_model[i].network
            model_json = model.to_json()
            with open(args.weight_path + args.v +"/model" + str(i) + ".json", "w") as json_file:
                json_file.write(model_json)
    multiagent.fit(args.num_iterations, args.eval_num, args.max_episode_length)

if __name__ == '__main__':
    main()
