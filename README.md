# ms-detect
Multiple sclerosis detection from MRI images.

#### Dataset
 - Dataset from [eHealth lab](http://www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets) of Department of Computer Science at University of Cyprus.

#### MSWavelet ([ipynb](./MSWavelet.ipynb))
 - Applies continuous or discrete wavelet transformations to MRI images.
 - Important for preprocessing.

#### CNNModel ([ipynb](./CNNModel.ipynb))
 - A convolutional neural network that predicts whether the input image is healthy or unhealthy.

#### Results
val_loss: 0.4371 - val_accuracy: 0.8351 - val_mse: 0.1271 - val_mae: 0.2133
