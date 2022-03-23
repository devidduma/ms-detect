# ms-detect
Multiple sclerosis detection from MRI images.

#### Dataset
 - Dataset from [eHealth lab](http://www.medinfo.cs.ucy.ac.cy/index.php/facilities/32-software/218-datasets) of Department of Computer Science at University of Cyprus.

#### Recipe
 - Pre-processing: apply continuous or discrete wavelet transformations to MRI images.
 - Binary classification with CNN: predict whether the input image is healthy or unhealthy using CNN with AlexNet or SqueezeNet macro-architectures.

#### Presentations
 - Wavelet Transformations: ([ipynb](presentations/preprocessing/mswavelet_presentation.ipynb))
 - Batch Wavelet Transformations: ([ipynb](presentations/preprocessing/dataset_preprocessing_batch.ipynb))
 - AlexNet results: ([AlexNet ipynb](presentations/cnn/alexnet_results.ipynb))
 - SqueezeNet results: ([SqueezeNet ipynb](presentations/cnn/squeezenet_results.ipynb))

#### Results
val_loss: 0.4376 - val_accuracy: 0.8297 - val_mse: 0.1396 - val_mae: 0.2852