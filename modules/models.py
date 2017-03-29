from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import RepeatVector
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class Models:
    def create_model(self):
        pass

class LinearModel(Models):

    def __init__(self, channels, input_shape, num_actions, model_name="linear"):
        self.channels = channels
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model_name = model_name

    def create_model(self):
        state_input = Input(shape=(self.channels * self.input_shape[0] * self.input_shape[1],), name='state_input')
        action_mask = Input(shape=(self.num_actions,), name='action_mask')

        dense1 = Dense(512, activation='sigmoid')(state_input)
        dense2 = Dense(64, activation='sigmoid')(dense1)

        action_output = Dense(self.num_actions, activation='linear', name='action_output')(dense2)
        masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')
        model = Model(input=[state_input, action_mask], output=masked_output)
        return model


class StanfordModel(Models):

    def __init__(self, channels, input_shape, num_actions, model_name="stanford"):
        self.channels = channels
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model_name = model_name

    def create_model(self):
        img_dims = (self.input_shape[0], self.input_shape[1], self.channels)
        state_input = Input(shape=img_dims, name='state_input')
        action_mask = Input(shape=(self.num_actions,), name='action_mask')

        conv1 = Convolution2D(32, 3, 3, \
                              border_mode='same', subsample=(1, 1))(state_input)

        conv2 = Convolution2D(32, 3, 3, activation='relu', \
                              border_mode='same', subsample=(1, 1))(conv1)

        conv2_norm = BatchNormalization()(conv2)

        conv3 = Convolution2D(32, 3, 3, \
                              border_mode='same', subsample=(1, 1))(conv2_norm)
        conv3_skip = merge([conv1, conv3], mode='sum')
        conv3_pool = MaxPooling2D(pool_size=(2, 2))(conv3_skip)

        conv4 = Convolution2D(32, 3, 3, activation='relu', \
                              border_mode='same', subsample=(1, 1))(conv3_pool)
        conv4_norm = BatchNormalization()(conv4)

        conv5 = Convolution2D(32, 3, 3, activation='relu', \
                              border_mode='same', subsample=(1, 1))(conv4_norm)
        conv5_skip = merge([conv3_pool, conv5], mode='sum')
        flatten = Flatten()(conv5_skip)
        dense_layer = Dense(512, activation='relu')(flatten)
        action_output = Dense(self.num_actions, activation='linear', name='action_output')(dense_layer)
        masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')
        model = Model(input=[state_input, action_mask], output=masked_output)
        return model


class DeepQModel(Models):

    def __init__(self, channels, input_shape, num_actions, model_name="deep"):
        self.channels = channels
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model_name = model_name

    def create_model(self):
        # 10 x 10 x channels (3 or 4)
        img_dims = (self.input_shape[0], self.input_shape[1], self.channels)
        state_input = Input(shape=img_dims, name='state_input')
        action_mask = Input(shape=(self.num_actions,), name='action_mask')

        conv = Convolution2D(32, 2, 2, activation='relu',
                                   border_mode='same', subsample=(2,2))(state_input)

        flatten = Flatten()(conv)

        if "dueling" in self.model_name:
            value_stream = Dense(512, activation='relu')(flatten)

            advantage_stream = Dense(512, activation='relu')(flatten)
            value_out = Dense(1, activation='linear', name='action_output')(value_stream)
            advantage_out = Dense(self.num_actions, activation='linear', name='advantage_out')(advantage_stream)

            rep_value = RepeatVector(self.num_actions)(value_out)
            rep_value = Flatten()(rep_value)

            if self.model_name == "dueling_av":
                advan_merge = Lambda(lambda y: y - K.mean(y, keepdims=True), output_shape=(self.num_actions,))(advantage_out)
            elif self.model_name == "dueling_max":
                advan_merge = Lambda(lambda y: y - K.max(y, keepdims=True), output_shape=(self.num_actions,))(advantage_out)
            else:
                advan_merge = advantage_out

            merged_action = merge(inputs=[rep_value, advan_merge], mode='sum', name='merged_action')
            masked_output = merge([action_mask, merged_action], mode='mul', name='merged_output')

        else:
            dense_layer1 = Dense(256, activation='sigmoid')(flatten)
            dense_layer2 = Dense(64, activation='sigmoid')(dense_layer1)
            action_output = Dense(self.num_actions, activation='linear', name='action_output')(dense_layer2)
            masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')

        model = Model(input=[state_input, action_mask], output=masked_output)

        return model