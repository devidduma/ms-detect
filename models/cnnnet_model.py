from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization

class CNNNet:
    def __init__(self, inputs=(512, 512, 3), data_format="channels_last"):
        self.inputs = inputs
        self.data_format = data_format

        if data_format == "channels_last":
            self.axis = 3
        elif data_format == "channels_first":
            self.axis = 1
        else:
            raise Exception("Data format must either be \'channels_first\' or \'channels_last\'.")

        self.input_kernel_size = 12
    
    def model(self):
        cnn = Sequential([
                    Conv2D(4, kernel_size=self.input_kernel_size, strides=1, padding="valid", activation="selu",
                                           input_shape=self.inputs, data_format=self.data_format),
                    MaxPool2D(pool_size=(2, 2)),
                    SpatialDropout2D(rate=0.1),
                    BatchNormalization(),
                    Conv2D(8, kernel_size=4, strides=1, padding="valid", activation="selu"),
                    MaxPool2D(pool_size=(2, 2)),
                    BatchNormalization(),
                    Conv2D(4, kernel_size=4, strides=1, padding="valid", activation="selu"),
                    MaxPool2D(pool_size=(2, 2)),
                    BatchNormalization(),
                    Conv2D(2, kernel_size=4, strides=1, padding="valid", activation="selu"),
                    MaxPool2D(pool_size=(2, 2)),
                    BatchNormalization(),
                    Conv2D(1, kernel_size=4, strides=1, padding="valid", activation="selu"),
                    MaxPool2D(pool_size=(2, 2)),
                    BatchNormalization(),
                    Flatten(),
                    Dense(32, activation="relu"),
                    Dropout(0.1),
                    Dense(16, activation="relu"),
                    Dense(8, activation="relu"),
                    Dense(1, activation="sigmoid")
                ])

        return cnn