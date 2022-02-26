from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Concatenate
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense


class SqueezeNet:
    def __init__(self, num_channels=8, inputs=(512, 512, 3), data_format="channels_last"):
        self.num_channels = num_channels
        self.inputs = inputs
        self.data_format = data_format

        if data_format=="channels_last":
            self.axis = 3
        elif data_format=="channels_first":
            self.axis = 1
        else:
            raise Exception("Data format must either be \'channels_first\' or \'channels_last\'.")

        self.firemodule_counter = 1

    def fire_module(self, inputlayer):
        fire_squeeze = Convolution2D(
            self.num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='squeeze_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(inputlayer)
        fire_expand1 = Convolution2D(
            self.num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='same', name='expand1_fire' + str(self.firemodule_counter),
            data_format=self.data_format)(fire_squeeze)
        fire_expand2 = Convolution2D(
            self.num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
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

        input_img = Input(shape=self.inputs)

        conv1 = Convolution2D(
            2*self.num_channels, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
            strides=(2, 2), padding='same', name='conv1',
            data_format=self.data_format)(input_img)
        maxpool1 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool1',
            data_format=self.data_format)(conv1)

        self.firemodule_counter = 2

        merge2 = self.fire_module(inputlayer=maxpool1)

        merge3 = self.fire_module(inputlayer=merge2)
        skip_conn3 = Concatenate(name="skip_conn3", axis=self.axis)([merge3, merge2])

        merge4 = self.fire_module(inputlayer=skip_conn3)
        skip_conn4 = Concatenate(name="skip_conn4", axis=self.axis)([merge4, merge3])
        maxpool4 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool4',
            data_format=self.data_format)(skip_conn4)

        merge5 = self.fire_module(inputlayer=maxpool4)
        skip_conn5 = Concatenate(name="skip_conn5", axis=self.axis)([merge5, maxpool4])

        merge6 = self.fire_module(inputlayer=skip_conn5)
        skip_conn6 = Concatenate(name="skip_conn6", axis=self.axis)([merge6, merge5])

        merge7 = self.fire_module(inputlayer=skip_conn6)
        skip_conn7 = Concatenate(name="skip_conn7", axis=self.axis)([merge7, merge6])

        merge8 = self.fire_module(inputlayer=skip_conn7)
        skip_conn8 = Concatenate(name="skip_conn8", axis=self.axis)([merge8, merge7])
        maxpool8 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool8',
            data_format=self.data_format)(skip_conn8)

        merge9 = self.fire_module(inputlayer=maxpool8)
        skip_conn9 = Concatenate(name="skip_conn9", axis=self.axis)([merge9, maxpool8])

        merge10 = self.fire_module(inputlayer=skip_conn9)
        skip_conn10 = Concatenate(name="skip_conn10", axis=self.axis)([merge10, merge9])
        maxpool10 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool10',
            data_format=self.data_format)(skip_conn10)

        merge11 = self.fire_module(inputlayer=maxpool10)
        skip_conn11 = Concatenate(name="skip_conn11", axis=self.axis)([merge11, maxpool10])

        merge12 = self.fire_module(inputlayer=skip_conn11)
        maxpool12 = MaxPooling2D(
            pool_size=(3, 3), strides=(2, 2), name='maxpool12',
            data_format=self.data_format)(merge12)

        """
        #dropout10 = Dropout(0.5, name='dropout10')(maxpool10)
        conv11 = Convolution2D(
            1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
            padding='valid', name='conv11',
            data_format=data_format)(dropout10)
        """
        #global_avgpool11 = GlobalAveragePooling2D(data_format=data_format)(maxpool10)

        flatten = Flatten(name="flatten", data_format=self.data_format)(maxpool12)
        sigmoid = Dense(1, name="sigmoid", activation="sigmoid")(flatten)

        return Model(inputs=input_img, outputs=sigmoid)