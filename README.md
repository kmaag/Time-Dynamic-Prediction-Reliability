# Time-Dynamic-Prediction-Reliability

Time-Dynamic Prediction Reliability is a post-processing tool for quantifying the reliability of neural networks (NNs) for semantic segmentation which can be easily added on top every NN. Time-Dynamic Prediction Reliability is a method that treats NNs like blackboxes, i.e. only using the NNs' softmax output. Based on that output, different aggregated dispersion measures are derived at segment level in order to fit another, preferably low complex and interpretable, model that indicates the prediction uncertainty per segment. As input for the models the dispersion measures of a segment in the considered frame serve, as well as the measures from the previous frames for this segment. For that a tracking algorithm is used beforehand, which identifies segments with each other over time. For each segment/connected component in the segmentation mask, Time-Dynamic Prediction Reliability, on the one hand, provides a method predicting whether this particular component intersects with the ground truth or not. Using the Intersection over Union (IoU) as performance measure, this latter task can be understood as "meta" classifying between the two classes {IoU=0} and {IoU>0}. On the other hand, Time-Dynamic Prediction Reliability ultimately also provides a method for quantifying the uncertainty of each predicted segment by predicting IoU values via regression, i.e. rating how well (accoding to IoU) each segment is predicted.

For further reading, please refer to https://arxiv.org/abs/1911.05075.

# Preparation:
We assume that the user is already using a neural network for semantic segmentation and a corresponding dataset. For each image from the segmentation dataset, Time-Dynamic Prediction Reliability requires a hdf5 file that contains the following data:

- a three-dimensional numpy array (height, width, classes) that contains the softmax probabilities computed for the current image
- the full filepath of the current input image
- a two-dimensional numpy array (height, width) that contains the ground truth class indices for the current image

Before running Time-Dynamic Prediction Reliability, please edit all necessary paths stored in "defs_global.py". The code is CPU based and parts of of the code trivially parallize over the number of input images, adjust "NUM_CORES" in "defs_global.py" to make use of this. Also, in the same file, select the tasks to be executed by setting the corresponding boolean variable (True/False).

# Run Code:
```sh
./x.sh
```

# Deeplabv3+, KITTI and VIPER:
The results in https://arxiv.org/abs/1911.05075 have been obtained from two Deeplabv3+ networks (https://github.com/tensorflow/models/tree/master/research/deeplab) together with the KITTI dataset (http://www.cvlibs.net/datasets/kitti/) and VIPER dataset (https://playing-for-benchmarks.org/overview/). 


# Authors:
Kira Maag (University of Wuppertal), Matthias Rottmann (University of Wuppertal)
