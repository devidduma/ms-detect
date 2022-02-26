from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Concatenate
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense


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
            num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='squeeze_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(inputlayer)
        fire_expand1 = Convolution2D(
            num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='expand1_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(fire_squeeze)
        fire_expand2 = Convolution2D(
            num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='expand2_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(fire_squeeze)
        merge = Concatenate(name="merge" + str(self.firemodule_counter),
                            axis=self.axis)([fire_expand1, fire_expand2])

        self.firemodule_counter += 1
        return merge

    def model(self):
        """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
        @param num_classes: total number of final categories
        Arguments:
        inputs -- shape of the input images (channel, cols, rows)
        """

        strides=(3, 3)

        input_img = Input(shape=self.inputs)

        conv1 = Convolution2D(
            16, (9, 9), activation='relu', kernel_initializer='glorot_uniform',
            strides=(1, 1), padding='same', name='conv1',
            data_format=self.data_format)(input_img)
        maxpool1 = MaxPooling2D(
            pool_size=(3, 3), strides=strides, name='maxpool1',
            data_format=self.data_format)(conv1)

        self.firemodule_counter = 2

        merge2 = self.fire_module(inputlayer=maxpool1, num_channels=16)
        merge3 = self.fire_module(inputlayer=merge2, num_channels=16)
        skip_conn3 = Concatenate(name="skip_conn3", axis=self.axis)([merge3, merge2])
        maxpool3 = MaxPooling2D(
            pool_size=(3, 3), strides=strides, name='maxpool3',
            data_format=self.data_format)(skip_conn3)

        merge4 = self.fire_module(inputlayer=maxpool3, num_channels=16)
        merge5 = self.fire_module(inputlayer=merge4, num_channels=16)
        skip_conn5 = Concatenate(name="skip_conn5", axis=self.axis)([merge5, merge4])
        maxpool5 = MaxPooling2D(
            pool_size=(3, 3), strides=strides, name='maxpool5',
            data_format=self.data_format)(skip_conn5)

        merge6 = self.fire_module(inputlayer=maxpool5, num_channels=16)
        merge7 = self.fire_module(inputlayer=merge6, num_channels=16)
        skip_conn7 = Concatenate(name="skip_conn7", axis=self.axis)([merge7, merge6])
        maxpool7 = MaxPooling2D(
            pool_size=(3, 3), strides=strides, name='maxpool7',
            data_format=self.data_format)(skip_conn5)

        merge8 = self.fire_module(inputlayer=maxpool7, num_channels=16)
        merge9 = self.fire_module(inputlayer=merge8, num_channels=16)
        skip_conn9 = Concatenate(name="skip_conn9", axis=self.axis)([merge9, merge8])
        maxpool9 = MaxPooling2D(
            pool_size=(3, 3), strides=strides, name='maxpool9',
            data_format=self.data_format)(skip_conn9)

        """
        #dropout10 = Dropout(0.5, name='dropout10')(maxpool10)
        conv11 = Convolution2D(
            1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='valid', name='conv11',
            data_format=data_format)(dropout10)
        """
        global_avgpool10 = GlobalAveragePooling2D(data_format=self.data_format)(maxpool9)
        sigmoid = Dense(1, name="sigmoid", activation="sigmoid")(global_avgpool10)

        return Model(inputs=input_img, outputs=sigmoid)