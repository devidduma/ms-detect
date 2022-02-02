from mswavelet import MSWavelet
import numpy as np
import tensorflow as tf
import sklearn
import sklearn.model_selection
import os
import pandas as pd
import matplotlib.pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_epoch, loss):
        self.min_epoch = min_epoch
        self.loss = loss

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.min_epoch and logs.get('loss') <= self.loss:
            self.model.stop_training = True


class CNNModel:
    def __init__(self, rootdir="MRIFreeDataset"):
        # Hyperparameters
        self.learningrate = 1e-3
        self.minibatchsize = 1
        self.epochs = 20
        self.callbacks = []

        self.rootdir = rootdir

        # define the train and val splits
        self.trainsplit = 0.8

        # Filename arrays: feeds the dataset
        self.filenames = []
        self.unhealthyfilenames = []
        self.healthyfilenames = []
        self.y = []

        # Dataset
        self.Xtrain_fn, self.Xtest_fn, self.ytrain, self.ytest = [], [], [], []

        # objects
        self.msw = MSWavelet(rootdir=rootdir)
        self.cnn = None

        # saves RAM, makes everything faster
        self.WT_DTYPE = np.float16

        # Upsampled images
        self.image_size = 512
        self.input_kernel_size = 12

        # Input channels
        self.inputchannels = 3
        self.color_mode = "rgb"

        # Print information
        print("CNNModel hyperparameters: ")
        print("Learning rate: ", self.learningrate)
        print("Minibatch size: ", self.minibatchsize)
        print("Epochs: ", self.epochs)
        print("Train/Test split: ", self.trainsplit)
        print("")

    def build_model(self):
        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(4, kernel_size=self.input_kernel_size, strides=1, padding="valid", activation="relu",
                                   input_shape=(self.image_size, self.image_size, self.inputchannels), data_format="channels_last"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=0.1),tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(8, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(4, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(2, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        cnn.summary()
        cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningrate), loss="binary_crossentropy",
                    metrics=["accuracy", "mse", 'mae'])

        self.cnn = cnn
        return cnn

    def pop_arrays_simple(self):
        self.unhealthyfilenames = self.msw.unhealthy_images
        self.healthyfilenames = self.msw.healthy_images
        self.filenames = self.unhealthyfilenames + self.healthyfilenames

        self.filenames = self.__clean_array__(self.filenames)
        self.unhealthyfilenames = self.__clean_array__(self.unhealthyfilenames)
        self.healthyfilenames = self.__clean_array__(self.healthyfilenames)

        self.y = [str(1)]*len(self.unhealthyfilenames) + [str(0)]*len(self.healthyfilenames)

        # info
        print("CNNModel arrays: ")
        print("self.filenames length: ", len(self.filenames))
        print("self.unhealthyfilenames length: ", len(self.unhealthyfilenames))
        print("self.healthyfilenames length: ", len(self.healthyfilenames))
        print("")

    def __clean_array__(self, array):
        result = []
        for fname in array:
            try:
                self.msw.tif_filename(filename=fname)
            except Exception as e:
                continue

            result.append(fname[len(self.rootdir) + 1:])
        return result

    # Helper for training
    def image_data_generator(self, augment=False):
        if augment:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=5,
                horizontal_flip=True,
                data_format="channels_last",
                dtype=self.WT_DTYPE
            )
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                data_format="channels_last",
                dtype=self.WT_DTYPE
            )

        return datagen

    def train_model(self):
        # Split into train and test sets
        self.Xtrain_fn, self.Xtest_fn, self.ytrain, self.ytest = sklearn.model_selection.train_test_split(self.filenames, self.y, stratify=self.y, random_state=42, train_size=self.trainsplit)

        Xtrain_fn = self.Xtrain_fn
        ytrain = self.ytrain
        Xtest_fn = self.Xtest_fn
        ytest = self.ytest

        dataframe = pd.DataFrame({
            "filename" : Xtrain_fn,
            "class" : ytrain
        })
        datagen = self.image_data_generator(augment=True)
        genflow = datagen.flow_from_dataframe(dataframe, directory=self.rootdir,
                    target_size=(self.image_size, self.image_size), color_mode=self.color_mode,
                    batch_size=self.minibatchsize, class_mode="binary", validate_filenames=False)

        dataframe_test = pd.DataFrame({
            "filename" : Xtest_fn,
            "class" : ytest
        })
        datagen_test = self.image_data_generator(augment=False)
        genflow_test = datagen_test.flow_from_dataframe(dataframe_test, directory=self.rootdir,
                    target_size=(self.image_size, self.image_size), color_mode=self.color_mode,
                    batch_size=self.minibatchsize, class_mode="binary", validate_filenames=False)

        H = self.cnn.fit(genflow, epochs=self.epochs, verbose=True, callbacks=self.callbacks,
                         validation_data=genflow_test)

        print("\n")

        # Save model
        self.cnn.save(os.path.join(self.msw.rootdir, 'model.h5'))
        print("Training finished! Model saved at \"model.h5\".")

        print("\n\n")

        # plot the training loss and accuracy
        N = self.epochs
        plt.style.use("fivethirtyeight")
        fig = plt.figure()
        fig.set_figwidth(8)
        fig.set_figheight(5)
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Plot of training loss and accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(self.rootdir, 'TrainingCurve.png'))
        plt.show()

    def test_model(self):
        Xtest_fn = self.Xtest_fn
        ytest = self.ytest

        dataframe = pd.DataFrame({
            "filename" : Xtest_fn,
            "class" : ytest
        })

        datagen = self.image_data_generator(augment=False)
        genflow = datagen.flow_from_dataframe(dataframe, directory=self.rootdir,
                    target_size=(self.image_size, self.image_size), color_mode=self.color_mode,
                    batch_size=self.minibatchsize, class_mode="binary", validate_filenames=False,
                    shuffle=False)
        scores = self.cnn.predict(genflow, verbose=True)

        scores = np.squeeze(scores, axis=1)
        ypred_labels = [str(int(x)) for x in np.rint(scores)]
        labels = ["0", "1"]

        print("\n\n")
        cr = sklearn.metrics.classification_report(ytest, ypred_labels, labels=labels)
        cm = sklearn.metrics.confusion_matrix(ytest, ypred_labels, labels=labels)

        print(cr, "\n")
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.get_cmap("Blues"))
        fig = plt.figure()
        fig.set_figwidth(6)
        fig.set_figheight(6)
        plt.savefig(os.path.join(self.rootdir, 'ConfusionMatrix.png'))
        plt.show()

if __name__ == '__main__':
    cnnmodel = CNNModel(rootdir="./Preprocessed")
    cnnmodel.pop_arrays_simple()

    #cnnmodel.build_model()
    #cnnmodel.train_model()