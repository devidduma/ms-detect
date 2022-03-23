from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Concatenate
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization


class SqueezeNet:
    def __init__(self, inputs=(512, 512, 3), data_format="channels_last"):
        self.inputs = inputs
        self.data_format = data_format

        if data_format=="channels_last":
            self.axis = 3
        elif data_format=="channels_first":
            self.axis = 1
        else:
            raise Exception("Data format must either be \'channels_first\' or \'channels_last\'.")

        self.firemodule_counter = 1

    def fire_module(self, inputlayer, num_channels):
        fire_squeeze = Convolution2D(
            num_channels/2, (1, 1), activation='selu', kernel_initializer='glorot_uniform',
            padding='same', name='squeeze_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(inputlayer)
        fire_squeeze = BatchNormalization()(fire_squeeze)

        fire_expand1 = Convolution2D(
            num_channels/2, (1, 1), activation='selu', kernel_initializer='glorot_uniform',
            padding='same', name='expand1_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(fire_squeeze)
        fire_expand1 = BatchNormalization()(fire_expand1)

        fire_expand2 = Convolution2D(
            num_channels/2, (3, 3), activation='selu', kernel_initializer='glorot_uniform',
            padding='same', name='expand2_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(fire_squeeze)
        fire_expand2 = BatchNormalization()(fire_expand2)

        merge = Concatenate(name="merge" + str(self.firemodule_counter),
                            axis=self.axis)([fire_expand1, fire_expand2])

        self.firemodule_counter += 1
        return merge

    def model(self):
        """
        Inspired by SqueezeNet(arXiv 1602.07360):
        but with less channels, bigger strides, SELU activation
        for binary classification.
        Macroarchitecture
        """

        strides=(3, 3)
        pool_size=(5, 5)

        input_img = Input(shape=self.inputs)

        conv1 = Convolution2D(
            8, (7, 7), activation='selu', kernel_initializer='glorot_uniform',
            strides=(2, 2), padding='same', name='conv1',
            data_format=self.data_format)(input_img)
        maxpool1 = MaxPooling2D(
            pool_size=pool_size, strides=strides, name='maxpool1',
            data_format=self.data_format)(conv1)
        maxpool1 = BatchNormalization()(maxpool1)

        self.firemodule_counter = 2

        merge2 = self.fire_module(inputlayer=maxpool1, num_channels=8)
        merge3 = self.fire_module(inputlayer=merge2, num_channels=8)
        skip_conn3 = Concatenate(name="skip_conn3", axis=self.axis)([merge3, merge2])

        merge4 = self.fire_module(inputlayer=skip_conn3, num_channels=8)
        maxpool4 = MaxPooling2D(
            pool_size=pool_size, strides=strides, name='maxpool4',
            data_format=self.data_format)(merge4)
        maxpool4 = BatchNormalization()(maxpool4)
        merge5 = self.fire_module(inputlayer=maxpool4, num_channels=8)
        skip_conn5 = Concatenate(name="skip_conn5", axis=self.axis)([merge5, maxpool4])

        merge6 = self.fire_module(inputlayer=skip_conn5, num_channels=8)
        merge7 = self.fire_module(inputlayer=merge6, num_channels=8)
        skip_conn7 = Concatenate(name="skip_conn7", axis=self.axis)([merge7, merge6])

        merge8 = self.fire_module(inputlayer=skip_conn7, num_channels=8)
        maxpool8 = MaxPooling2D(
            pool_size=pool_size, strides=strides, name='maxpool8',
            data_format=self.data_format)(merge8)
        maxpool8 = BatchNormalization()(maxpool8)
        merge9 = self.fire_module(inputlayer=maxpool8, num_channels=8)
        skip_conn9 = Concatenate(name="skip_conn9", axis=self.axis)([merge9, maxpool8])

        conv_last = Convolution2D(
            1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            strides=(1, 1), padding='same', name='conv_last',
            data_format=self.data_format)(skip_conn9)
        conv_last = BatchNormalization()(conv_last)

        flatten = Flatten()(conv_last)
        dense = Dense(48, name="dense", activation="relu")(flatten)
        dropout = Dropout(0.2, name="dropout")(dense)
        dense2 = Dense(16, name="dense2", activation="relu")(dropout)

        sigmoid = Dense(1, name="sigmoid", activation="sigmoid")(dense2)

        return Model(inputs=input_img, outputs=sigmoid)