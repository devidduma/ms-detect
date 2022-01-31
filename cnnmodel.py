import PIL.Image
from mswavelet import MSWavelet
import numpy as np
import tensorflow as tf
import sklearn
import pywt
import gc
import os


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_epoch, loss):
        self.min_epoch = min_epoch
        self.loss = loss

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.min_epoch and logs.get('loss') <= self.loss:
            self.model.stop_training = True


""" Good combinations:
self.image_size = [512, 1024]
self.input_kernel_size = [12, 16]

hint: (512, 12) is a lot faster to train
"""
class CNNModel:
    def __init__(self, rootdir="./MRIFreeDataset/Initial & repeat MRI in MS-Free Dataset"):
        # Hyperparameters
        self.learningrate = 1e-3
        self.batchsize = 16
        self.minibatchsize = 2
        self.batchsize_test = 4
        self.epochs = 2
        self.callbacks = []
        self.data_augment = True

        # define the train and val splits
        self.trainsplit = 0.75

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
        self.DTYPE = np.uint8
        self.WT_DTYPE = np.float16

        # Upsampled images
        self.image_size = 512
        self.input_kernel_size = 12
        """ Good combinations:
        self.image_size = [512, 1024]
        self.input_kernel_size = [12, 16]
        """

        # Wavelet Transforms to apply
        self.dwtlist = []
        self.cwtlist = ["mexh", "fbsp", "cgau4"]
        self.cwtscales = [[8], [18], [16]]
        self.inputchannels = 4

        # Print information
        print("CNNModel hyperparameters: ")
        print("Learning rate: ", self.learningrate)
        print("Batch size: ", self.batchsize)
        print("Minibatch size: ", self.minibatchsize)
        print("Batch size for testing: ", self.batchsize_test)
        print("Epochs: ", self.epochs)
        print("Data augmentation: ", self.data_augment)
        print("Train/Test split: ", self.trainsplit)
        print("")

    def build_model(self):

        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=self.input_kernel_size, strides=1, padding="valid", activation="relu",
                                   input_shape=(self.image_size, self.image_size, self.inputchannels), data_format="channels_last"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=0.3),tf.keras.layers.BatchNormalization(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=0.25),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.SpatialDropout2D(rate=0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, kernel_size=4, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(4, kernel_size=3, strides=1, padding="valid", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
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
        # make sure every filename is a valid one
        self.unhealthyfilenames = self.__clean_array__(self.unhealthyfilenames)
        self.healthyfilenames = self.__clean_array__(self.healthyfilenames)

        # create filename array and y array for main dataset
        self.filenames = self.unhealthyfilenames + self.healthyfilenames
        self.y = [1]*len(self.unhealthyfilenames) + [0]*len(self.healthyfilenames)

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
            result.append(fname)
        return result

    # generate training samples
    def generateXproc(self, files, cwtlist, dwtlist, cwtscales, only_last_wt=True):

        # cast files to numpy array
        files = np.asarray(files, dtype=self.DTYPE)

        # List of results of Wavelet Transforms
        reswt = []

        # Continuous wavelet transforms
        for i in range(len(cwtlist)):
            res, _ = pywt.cwt(files, wavelet=cwtlist[i], scales=cwtscales[i])
            if only_last_wt:
                res = res[-1]
                res = np.array(res, dtype=self.WT_DTYPE)
                res = np.expand_dims(res, axis=0)
            else:
                res = np.array(res, dtype=self.WT_DTYPE)
            # append only last WT
            reswt.append(res)

        # Discrete wavelet transforms
        for i in range(len(dwtlist)):
            _, res = pywt.dwt2(files, wavelet=dwtlist[i])
            if only_last_wt:
                res = res[-1]
                res = np.array(res, dtype=self.WT_DTYPE)
                res = np.expand_dims(res, axis=0)
            else:
                res = np.array(res, dtype=self.WT_DTYPE)
            # Upsample back to self.image_size x self.image_size
            res = np.array(self.msw.upsample(res, size=self.image_size, info=False), dtype=self.WT_DTYPE)
            # append only last WT
            reswt.append(res)

        # Concatenate channels from wavelet transforms
        # add wavelet transforms
        res = np.array(np.concatenate(reswt, axis=0), dtype=self.WT_DTYPE)

        # Permute dimensions to (N,C,H,W)
        res = np.transpose(res, axes=(1, 0, 2, 3))

        print("Generated dataset shape: ", res.shape)
        return res

    def attach_imgs_to_dataset(self, dataset, img_batch):
        img_batch = np.asarray(img_batch)
        dataset = np.asarray(dataset)
        if img_batch.ndim != 3:
            raise Exception("Batch of images must have 3 dimensions.")
        if dataset.ndim != 4:
            raise Exception("Dataset must have 4 dimensions.")
        if len(img_batch) != len(dataset):
            raise Exception("Batch of images and dataset must have the same length.")

        dataset = np.transpose(dataset, axes=(1, 0, 2, 3))
        img_batch = np.expand_dims(img_batch, axis=0)
        dataset = np.append(img_batch, dataset, axis=0)
        dataset = np.transpose(dataset, axes=(1, 0, 2, 3))

        print("Dataset shape after attaching: ", dataset.shape)
        return dataset

    # Helper for training
    def image_generator(self, X, y):
        if self.data_augment:
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

        genflow = datagen.flow(X, y, batch_size=self.minibatchsize)
        return genflow

    def load_images(self, filenames):
        # result img batch
        result = np.empty(shape=(0, self.image_size, self.image_size), dtype=self.DTYPE)
        original_sizes = []

        for filename in filenames:
            filename = self.msw.filename_stripped(filename)
            # load image
            img = PIL.Image.open(self.msw.tif_filename(filename))
            # cast to numpy array
            filenp = np.array(img, dtype=self.DTYPE)
            # mark original size
            original_sizes.append(filenp.shape[-1])
            # upsample
            filenp = self.msw.upsample(filenp, size=self.image_size, info=False)
            filenp = np.expand_dims(filenp, axis=0)

            # add to result
            result = np.append(result, filenp, axis=0)

        print("Loaded image batch of size: ", result.shape)
        return result, original_sizes

    def img_with_lesions_batch(self, files, filenames, original_sizes):
        if np.asarray(files).ndim != 3:
            raise Exception("Files batch must have 3 dimensions.")
        if len(files) != len(filenames):
            raise Exception("Number of files should be equal to the number of filenames.")

        # result img batch
        result = np.empty(shape=(0, self.image_size, self.image_size))

        for i in range(len(files)):
            file = files[i]
            filename = filenames[i]
            original_size = original_sizes[i]
            file_with_lesion = self.msw.img_with_lesions(file, filename, original_size, self.DTYPE, self.image_size)
            file_with_lesion = np.array(file_with_lesion)
            file_with_lesion = np.expand_dims(file_with_lesion, axis=0)

            # add to results
            result = np.append(result, file_with_lesion, axis=0)

        print("File with lesions array shape: ", result.shape)
        return result

    def train_model(self):
        # free memory

        # Split into train and test sets
        self.Xtrain_fn, self.Xtest_fn, self.ytrain, self.ytest = sklearn.model_selection.train_test_split(self.filenames, self.y, train_size=self.trainsplit)

        # Train and keep track of avg loss and accuracy
        resloss = []
        resaccuracy = []
        resmse = []
        resmae = []

        for index in range(0, len(self.Xtrain_fn), self.batchsize):
            gc.collect()

            print("\nTraining index run at: ", index, "\n")

            # batch of filenames
            batch_filenames = self.Xtrain_fn[index:index + self.batchsize]
            # load batch of files
            batch_files, original_sizes = self.load_images(batch_filenames)
            # batch of files with lesions
            #batch_files_with_lesions = self.img_with_lesions_batch(batch_files, batch_filenames, original_sizes)

            # Dataset: wavelet transform
            Xbatch = self.generateXproc(batch_files,
                                        dwtlist=self.dwtlist, cwtlist=self.cwtlist, cwtscales=self.cwtscales)
            # Dataset: append original files as a channel
            Xbatch = self.attach_imgs_to_dataset(Xbatch, batch_files)
            # Y labels
            ybatch = self.ytrain[index:index + self.batchsize]

            # Permute dimensions to fit input shape
            Xbatch = np.transpose(Xbatch, axes=(0, 2, 3, 1))

            genflow = self.image_generator(Xbatch, ybatch)
            self.cnn.fit(genflow, epochs=self.epochs, verbose=True, callbacks=self.callbacks)

            del Xbatch
            del batch_files
            #del batch_files_with_lesions
            del genflow

        # Save model
        self.cnn.save(os.path.join(self.msw.rootdir, 'model.h5'))
        print("Training finished! Model saved at \"model.h5\".")

    def test_model(self):
        # Evaluate model
        scores = []
        for index in range(0, len(self.Xtest_fn), self.batchsize_test):
            gc.collect()

            print("\nTesting index run at: ", index, "\n")

            # batch of filenames
            batch_filenames = self.Xtest_fn[index:index + self.batchsize_test]
            # load batch of files
            batch_files, original_sizes = self.load_images(batch_filenames)

            # Dataset: wavelet transform
            Xbatchtest = self.generateXproc(batch_files,
                                        dwtlist=self.dwtlist, cwtlist=self.cwtlist, cwtscales=self.cwtscales)
            # Dataset: append original images as a channel
            Xbatchtest = self.attach_imgs_to_dataset(Xbatchtest, batch_files)
            # Y labels
            ybatchtest = self.ytest[index:index + self.batchsize_test]

            # Permute dimensions to fit input shape
            Xbatchtest = np.transpose(Xbatchtest, axes=(0, 2, 3, 1))

            # cast to numpy array
            ybatchtest = np.array(ybatchtest)
            score = self.cnn.evaluate(Xbatchtest, ybatchtest, verbose=True)
            scores.append(score)

            del batch_files
            del Xbatchtest

        # calculate loss and accuracy
        scores = np.array(scores)
        mean = np.mean(scores, axis=0)
        print("\nTest score results:")
        print('Mean loss: ', mean[0])
        print('Mean accuracy: ', mean[1])
        print('Mean MSE: ', mean[2])
        print('Mean MAE: ', mean[3])


if __name__ == '__main__':
    cnnmodel = CNNModel()
    cnnmodel.pop_arrays_simple()

    cnnmodel.build_model()
    #cnnmodel.train_model()