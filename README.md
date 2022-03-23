# ms-detect
Multiple sclerosis detection from MRI images.

#### Dataset
 - Dataset from [eHealth lab](http://www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets) of Department of Computer Science at University of Cyprus.

#### MSWavelet ([ipynb](presentations/preprocessing/mswavelet_presentation.ipynb))
 - Applies continuous or discrete wavelet transformations to MRI images.
 - Important for preprocessing.

#### CNN models ([AlexNet ipynb](presentations/cnn/alexnet_results.ipynb), [SqueezeNet ipynb](presentations/cnn/squeezenet_results.ipynb))
 - CNN models with AlexNet or SqueezeNet macro-architectures. Used to predict whether the input image is healthy or unhealthy.

#### Results
val_loss: 0.4376 - val_accuracy: 0.8297 - val_mse: 0.1396 - val_mae: 0.2852