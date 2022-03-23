from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input


class TL_EfficientNetB0:
    def __init__(self, inputs=(512, 512, 3)):
        self.inputs = inputs

    def model(self):
        cnn_base = EfficientNetB0(
            include_top=False, weights='imagenet',
            input_tensor=Input(shape=self.inputs)
        )

        for layer in cnn_base.layers[:]:
            layer.trainable = False

        cnn = cnn_base.output
        cnn = GlobalAveragePooling2D()(cnn)
        cnn = Dense(64, activation="relu")(cnn)
        cnn = Dense(16, activation="relu")(cnn)
        cnn = Dense(8, activation="relu")(cnn)
        cnn = Dense(1, activation="sigmoid")(cnn)
        cnn = Model(inputs=cnn_base.input, outputs=cnn)

        return cnn