import PIL.Image
from mswavelet import MSWavelet
import numpy as np
import tensorflow as tf
import sklearn
import sklearn.model_selection
import os
import pandas as pd
import matplotlib.pyplot as plt
from models.cnnnet_model import CNNNet
from models.transferlearning_efficientnet import TL_EfficientNetB0


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_epoch, loss):
        self.min_epoch = min_epoch
        self.loss = loss

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.min_epoch and logs.get('loss') <= self.loss:
            self.model.stop_training = True


class Main:
    def __init__(self, rootdir="MRIFreeDataset"):
        # Root directory
        self.rootdir = rootdir

        # Hyperparameters
        self.learningrate = 1e-3
        self.minibatchsize = 1
        self.epochs = 30

        # Model Checkpoint callback
        self.checkpoint_filepath = os.path.join(self.rootdir, 'best_model.h5')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
        # Callbacks
        self.callbacks = [model_checkpoint_callback]

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
        self.input_channels = 3
        self.color_mode = "rgb"

        # Print information
        print("CNNModel hyperparameters: ")
        print("Learning rate: ", self.learningrate)
        print("Minibatch size: ", self.minibatchsize)
        print("Epochs: ", self.epochs)
        print("Train/Test split: ", self.trainsplit)
        print("")

    def build_model_CNNNet(self):
        self.cnn = CNNNet(inputs=(self.image_size, self.image_size, self.input_channels)).model()
        self.cnn.summary()
        self.cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningrate), loss="binary_crossentropy",
                    metrics=["accuracy", "mse", "mae"])

    def transfer_learning_EfficientNetB0(self):
        self.cnn = TL_EfficientNetB0(inputs=(self.image_size, self.image_size, self.input_channels)).model()
        self.cnn.summary()
        self.cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learningrate), loss="binary_crossentropy",
                    metrics=["accuracy", "mse", "mae"])

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
                rotation_range=15,
                horizontal_flip=True,
                zoom_range=0.1,
                data_format="channels_last",
                dtype=self.WT_DTYPE
            )
        else:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                data_format="channels_last",
                dtype=self.WT_DTYPE
            )

        return datagen

    def train_model(self, augment=True):
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
        datagen = self.image_data_generator(augment=augment)
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

        #print("\n")
        # Save model:
        # The model is already saved by the ModelCheckpoint callback!
        # No need to do anything.

        print("\n\n")

        # plot the training loss and accuracy
        N = self.epochs
        plt.style.use("fivethirtyeight")
        fig = plt.figure()
        fig.set_figwidth(8)
        fig.set_figheight(5)
        plt.plot(np.arange(0, N), H.history["loss"], label="training loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="validation loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.rootdir, 'LossCurve.png'))
        plt.show()

        plt.clf()
        N = self.epochs
        plt.style.use("fivethirtyeight")
        fig = plt.figure()
        fig.set_figwidth(8)
        fig.set_figheight(5)
        plt.plot(np.arange(0, N), H.history["accuracy"], label="training accuracy")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="validation accuracy")
        plt.title("Training Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.rootdir, 'AccuracyCurve.png'))
        plt.show()

    def test_model(self):
        # Load best model
        self.cnn = tf.keras.models.load_model(self.checkpoint_filepath)

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
        self.cnn.evaluate(genflow, verbose=True)
        scores = self.cnn.predict(genflow, verbose=True)

        scores = np.squeeze(scores, axis=1)
        ypred_labels = [str(int(x)) for x in np.rint(scores)]
        labels = ["0", "1"]

        print("\n\n")
        cr = sklearn.metrics.classification_report(ytest, ypred_labels, labels=labels)
        cm = sklearn.metrics.confusion_matrix(ytest, ypred_labels, labels=labels)

        print(cr, "\n")
        fig = plt.figure()
        fig.set_figwidth(6)
        fig.set_figheight(6)
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.get_cmap("Blues"))
        plt.savefig(os.path.join(self.rootdir, 'ConfusionMatrix.png'))
        plt.show()

    def apply_data_augmentation(self, imgpath, input_dir="./MRIFreeDataset", output_dir="./data_augment"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        img = PIL.Image.open(os.path.join(input_dir, imgpath))
        imgexp = np.expand_dims(np.asarray(img), 0)
        imgexp = np.transpose(imgexp, axes=(1, 2, 0))
        imgexp = np.expand_dims(imgexp, 0)

        datagen = self.image_data_generator(augment=True)
        datagen.fit(imgexp)
        for x, val in zip(datagen.flow(imgexp, save_to_dir=output_dir, save_prefix="aug", save_format="png"),
                          range(10)):
            pass


if __name__ == '__main__':
    main = Main(rootdir="./Preprocessed")
    main.pop_arrays_simple()

    """
    main.apply_data_augmentation(imgpath=main.unhealthyfilenames[100], input_dir="./MRIFreeDataset", output_dir="./data_augment")

    main.transfer_learning_EfficientNetB0()
    main.train_model(augment=False)
    main.test_model()

    main.build_model_CNNNet()
    main.train_model(augment=True)
    main.test_model()
    """