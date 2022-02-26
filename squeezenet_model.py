from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Concatenate
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D


def SqueezeNet(nb_classes, inputs=(512, 512, 3), num_channels=16, data_format="channels_last"):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """

    if data_format=="channels_last":
        axis=3
    elif data_format=="channels_first":
        axis=1
    else:
        raise Exception("Data format must either be \'channels_first\' or \'channels_last\'.")

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        2*num_channels, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format=data_format)(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format=data_format)(conv1)
    fire2_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format=data_format)(maxpool1)
    fire2_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format=data_format)(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format=data_format)(fire2_squeeze)
    merge2 = Concatenate(name="merge2", axis=axis)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format=data_format)(merge2)
    fire3_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format=data_format)(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format=data_format)(fire3_squeeze)
    # merge + skip connections as input
    merge3 = Concatenate(name="merge3", axis=axis)([fire3_expand1, fire3_expand2])
    skip_conn3 = Concatenate(name="skip_conn3", axis=axis)([merge3, merge2])

    fire4_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format=data_format)(skip_conn3)
    fire4_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format=data_format)(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format=data_format)(fire4_squeeze)
    # merge + skip connection
    merge4 = Concatenate(name="merge4", axis=axis)([fire4_expand1, fire4_expand2])
    skip_conn4 = Concatenate(name="skip_conn4", axis=axis)([merge4, merge3])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format=data_format)(skip_conn4)

    fire5_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format=data_format)(maxpool4)
    fire5_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format=data_format)(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format=data_format)(fire5_squeeze)
    # merge + skip connections
    merge5 = Concatenate(name="merge5", axis=axis)([fire5_expand1, fire5_expand2])
    skip_conn5 = Concatenate(name="skip_conn5", axis=axis)([merge5, maxpool4])

    fire6_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format=data_format)(skip_conn5)
    fire6_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format=data_format)(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format=data_format)(fire6_squeeze)
    # merge + skip connections
    merge6 = Concatenate(name="merge6", axis=axis)([fire6_expand1, fire6_expand2])
    skip_conn6 = Concatenate(name="skip_conn6", axis=axis)([merge6, merge5])

    fire7_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format=data_format)(skip_conn6)
    fire7_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format=data_format)(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format=data_format)(fire7_squeeze)
    # merge + skip connections
    merge7 = Concatenate(name="merge7", axis=axis)([fire7_expand1, fire7_expand2])
    skip_conn7 = Concatenate(name="skip_conn7", axis=axis)([merge7, merge6])

    fire8_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format=data_format)(skip_conn7)
    fire8_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format=data_format)(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format=data_format)(fire8_squeeze)
    # merge + skip connections
    merge8 = Concatenate(name="merge8", axis=axis)([fire8_expand1, fire8_expand2])
    skip_conn8 = Concatenate(name="skip_conn8", axis=axis)([merge8, merge7])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format=data_format)(skip_conn8)
    fire9_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format=data_format)(maxpool8)
    fire9_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format=data_format)(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format=data_format)(fire9_squeeze)
    merge9 = Concatenate(name="merge9", axis=axis)([fire9_expand1, fire9_expand2])
    skip_conn9 = Concatenate(name="skip_conn9", axis=axis)([merge9, maxpool8])

    fire10_squeeze = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_squeeze',
        data_format=data_format)(skip_conn9)
    fire10_expand1 = Convolution2D(
        num_channels, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_expand1',
        data_format=data_format)(fire10_squeeze)
    fire10_expand2 = Convolution2D(
        num_channels, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire10_expand2',
        data_format=data_format)(fire10_expand1)
    merge10 = Concatenate(name="merge10", axis=axis)([fire10_expand1, fire10_expand2])
    skip_conn10 = Concatenate(name="skip_conn10", axis=axis)([merge10, merge9])
    maxpool10 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool10',
        data_format=data_format)(skip_conn10)

    #fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv11 = Convolution2D(
        1, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='valid', name='conv11',
        data_format=data_format)(maxpool10)
    global_avgpool11 = GlobalAveragePooling2D(data_format=data_format)(conv11)
    sigmoid = Activation("sigmoid", name='sigmoid')(global_avgpool11)

    return Model(inputs=input_img, outputs=sigmoid)