#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym
import numpy as np
import tensorflow as tf

from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.optimizers import RMSprop, SGD, Adam, Nadam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from modules.dqn_agent import DQNAgent
from modules.objectives import mean_huber_loss, huber_loss
from modules.utils import argrender

import envs.pacman_envs

def create_model(channels, input_shape, num_actions,
                 model_name='linear'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understand your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    channels: int
      Each input to the network has channels
    input_shape: tuple(int, int)
      The expected input image size.
    self.num_actions: int
      Number of possible actions. Defined by the gym environment.
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    # instantiate model

    if(model_name=='linear'):
        return linear_q_model(channels, input_shape, num_actions)

    elif(model_name=='deep' or 'dueling' in model_name):
        return deep_q_model(channels, input_shape, num_actions, model_name)
    else:
        raise Exception("Unknown Q Model Name: Try linear_q or deep_q")

def linear_q_model(channels, input_shape, num_actions):
    state_input = Input(shape=(channels*input_shape[0]*input_shape[1], ), name='state_input')

    action_mask = Input(shape=(self.num_actions, ), name='action_mask')

    state_input_norm = BatchNormalization(mode=0, axis=-1)(state_input)

    action_output = Dense(num_actions, activation='linear', name='action_output')(state_input_norm)

    masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')

    model = Model(input=[state_input, action_mask], output=masked_output)

    return Model(input=[state_input, action_mask], output=masked_output)

def deep_q_model(channels, input_shape, num_actions, model_name):
    # 10 x 10 x 3
    img_dims =(input_shape[0], input_shape[1], channels)

    state_input = Input(shape=img_dims, name='state_input')

    action_mask = Input(shape=(num_actions, ), name='action_mask')

    state_input_norm = BatchNormalization()(state_input)

    #The first hidden layer convolves 32 8 x 8 filters with stride 4
    first_conv = Convolution2D(32, 8, 8, activation='relu', \
        border_mode='same', subsample=(4, 4))(state_input_norm)

    # convolves 64 4 x 4 filters with stride 2
    second_conv = Convolution2D(64, 4, 4, activation='relu', \
        border_mode='same', subsample=(2, 2))(first_conv)

    # convolves 64 3 x 3 filters with stride 1
    third_conv = Convolution2D(64, 3, 3, activation='relu', \
        border_mode='same', subsample=(1, 1))(second_conv)

    flatten = Flatten()(third_conv)

    if  "dueling" in model_name:
        value_stream = Dense(512, activation='relu')(flatten)

        advantage_stream = Dense(512, activation='relu')(flatten)

        value_out =  Dense(1, activation='linear', name='action_output')(value_stream)

        advantage_out = Dense(num_actions, activation='linear', name='advantage_out')(advantage_stream)

        rep_value = RepeatVector(num_actions)(value_out)
        rep_value = Flatten()(rep_value)

        if model_name == "dueling_av":
            advan_merge = Lambda(lambda y: y - K.mean(y, keepdims=True), output_shape=(num_actions,))(advantage_out)
        elif model_name == "dueling_max":
            advan_merge = Lambda(lambda y: y - K.max(y, keepdims=True), output_shape=(num_actions,))(advantage_out)
        else:
            advan_merge = advantage_out

        merged_action = merge(inputs=[rep_value, advan_merge],  mode='sum', name='merged_action')

        masked_output = merge([action_mask, merged_action], mode='mul', name='merged_output')
    else:
        dense_layer = Dense(512, activation='relu')(flatten)

        action_output = Dense(num_actions, activation='linear', name='action_output')(dense_layer)

        masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')

    model = Model(input=[state_input, action_mask], output=masked_output)

    return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def make_assertions(args):
    algo = args.algorithm
    network_name = args.network_name
    assert network_name == 'linear' or network_name == 'deep' or 'dueling' in network_name
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
    parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial epsilon pre-decay')
    parser.add_argument('--loss', default='mean_huber', help='mean_huber, huber, mae, or mse.')
    parser.add_argument('--lr', default=0.00025, type=float, help='(initial) learning rate')
    parser.add_argument('--max_episode_length', default=1000, type=int, help='Max episode length (for training, not eval).')
    parser.add_argument('--memory', default=1e6, type=int, help='size of buffer for experience replay')
    parser.add_argument('--momentum', default=0.95, type=float)
    parser.add_argument('--network_name', default='deep', help='Model Name: deep or linear or dueling or dueling_av or dueling_max')
    parser.add_argument('--optimizer', default='adam', help='one of sgd, rmsprop, and adam')
    parser.add_argument('--num_burn_in', default=5e4, type=int, help='Buffer size pre-training.')
    parser.add_argument('--num_decay_steps', default=1e6, type=int, help='Epsilon policy decay length')
    parser.add_argument('--num_iterations', default=5e7, type=int, help='Number frames visited for training.')
    parser.add_argument('--target_update_freq', default=1e4, type=int, help='Target Update frequency. Only applies to algorithm==replay_target, double, dueling.')
    parser.add_argument('--update_freq', default=4, type=int, help='Update frequency.')
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

    # optimizer
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

    model = create_model(args.channels, (args.dim, args.dim), args.num_actions, model_name=args.network_name)
    model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
    if args.verbose == 2:
        model.summary()
        argrender(args)

    target_network = None
    if args.algorithm == 'double' and args.network_name == 'linear':
        target_network = create_model(args.channels, (args.dim, args.dim), args.num_actions, model_name=args.network_name)
        target_network.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

    # create your agent and run it
    model = DQNAgent(model, target_network, args)

    if bool(args.eval_random):
        model.evaluate_random(env)

    model.fit(env, args.num_iterations, args.eval_num, args.max_episode_length)

if __name__ == '__main__':
    main()
