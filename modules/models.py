from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import RepeatVector
from keras.layers.merge import Concatenate, Multiply, Add
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import backend as K

class Models:
    def create_model(self):
        pass

class LinearStreamModel(Models):
    def __init__(self, input_shape, activation, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.activation = activation

    def create_model(self):
        state_input = Input(shape=(2,), name='state_input')
        action_mask = Input(shape=(self.num_actions,), name='action_mask')

        other_state_input = Input(shape=(2,), name='other_state_input')

        upscale_me = Dense(8, activation=self.activation)(state_input)
        upscale_other = Dense(8, activation=self.activation)(other_state_input)

        merged = Concatenate(axis=-1)([upscale_me, upscale_other])

        joint_rep = Dense(8, activation=self.activation)(merged)

        # _, 8
        skip_connection = Add()([upscale_me, joint_rep])

        action_output = Dense(self.num_actions, activation='linear', name='action_output')(skip_connection)
        masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')

        return Model(input=[state_input, other_state_input, action_mask], output=masked_output)

class LinearModel(Models):

    def __init__(self, input_shape, activation, num_actions, model_name):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.activation = activation

        model_name_split = model_name.split(':')
        self.model_name = model_name_split[0]

        if len(model_name_split) == 2:
            self.dueling = model_name_split[1]
        else:
            self.dueling = "no"

    def create_model(self):
        state_input = Input(shape=(4,), name='state_input')
        action_mask = Input(shape=(self.num_actions,), name='action_mask')

        dense1 = Dense(16, activation=self.activation)(state_input)

        if "dueling" in self.model_name:
            value_stream = Dense(128, activation='relu')(dense2)

            advantage_stream = Dense(128, activation='relu')(dense2)
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
            action_mask = Input(shape=(self.num_actions,), name='action_mask')
            merged_action = merge(inputs=[rep_value, advan_merge], mode='sum', name='merged_action')
            masked_output = merge([action_mask, merged_action], mode='mul', name='merged_output')

        else:
            dense2 = Dense(16, activation=self.activation)(dense1)
            dense3 = Dense(16, activation=self.activation)(dense2)
            action_output = Dense(self.num_actions, activation='linear', name='action_output')(dense3)
            masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')

        model = Model(input=[state_input, action_mask], output=masked_output)
        return model

class StanfordModel(Models):

    def __init__(self, input_shape, num_actions, model_name="stanford"):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model_name = model_name

    def create_model(self):
        img_dims = (self.input_shape[0], self.input_shape[1])
        state_input = Input(shape=img_dims, name='state_input')

        # state_input_padded = ZeroPadding2D((1,1))(state_input)

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

    def __init__(self, input_shape, num_actions, model_name="deep"):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model_name = model_name

    def create_model(self):
        # 10 x 10 x channels (1)
        img_dims = (self.input_shape[0], self.input_shape[1], 1)
        state_input = Input(shape=img_dims, name='state_input')


        conv = Convolution2D(32, 3, 3, activation='relu',
                                   border_mode='same', subsample=(1, 1))(state_input)

        if self.input_shape[0] > 3:
            conv = Convolution2D(32, 3, 3, activation='relu', border_mode='same', subsample=(1,1))(conv)

        conv2 = Convolution2D(32, 2, 2, activation='relu',
            border_mode='same', subsample=(1, 1))(conv)
        flatten = Flatten()(conv2)

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
            action_mask = Input(shape=(self.num_actions,), name='action_mask')
            merged_action = merge(inputs=[rep_value, advan_merge], mode='sum', name='merged_action')
            masked_output = merge([action_mask, merged_action], mode='mul', name='merged_output')
        else:
            dense_layer1 = Dense(512, activation='relu')(flatten)
            dropout1 = Dropout(0.5)(dense_layer1)
            dense_layer2 = Dense(256, activation='relu')(dropout1)
            dropout2 = Dropout(0.5)(dense_layer2)
            if "combo" in self.model_name:
                action_mask = Input(shape=(self.num_actions * self.num_actions,), name='action_mask')
                action_output = Dense(self.num_actions * self.num_actions, activation='linear', name='action_output')(dense_layer1)
                masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')
            else:
                action_mask = Input(shape=(self.num_actions,), name='action_mask')
                action_output = Dense(self.num_actions, activation='linear', name='action_output')(dense_layer1)
                masked_output = merge([action_mask, action_output], mode='mul', name='merged_output')


        model = Model(input=[state_input, action_mask], output=masked_output)

        model.summary()

        return model