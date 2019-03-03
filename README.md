# Outline
This is real-time face detection and classification of the detected face using a single fully Convolutional Neural Network. The aim of the project was to experiment with combining regression and classification and have a model do both simultaneously. The bounding boxes are produced using regression and standard classification is used for classifying the detected objects. 

The aim of the project was `NOT` to focus produce highly accurate models this can also be seen in results as the model failed to learn the age group classification although it does work for gender classification and face detection. It was only an experiment and also a learning exercise for me as it uses a custom NN with a custom loss function. It doesn't follow any popular architecture. 

# Model Overview
The model tries simultaneously detect as well as classify faces using a single NN. It directly takes an image and there is no preprocessing of input. It directly outputs the bounding boxes with confidence score as well as probabilities for different classes. The only post-processing step is [Non-Max suppression][non-max]. 

Similar to [YOLO][yolo] NN, the model considers the input image as a grid and produces a bounding box and conditional probabilities per grid cell. That is the only concept I borrowed from YOLO. It doesn't use YOLO's loss function or any pre-trained model. The loss function is simply the sum of average loss for individual components.

# Dataset
I used a part of [IMDB-Wiki face dataset] for training and testing the models. The data was preprocessed before use. All the details are [here][preprocess].

# Performance

- Face detection IOU: 0.634
- Gender Accuracy: 0.781
- Age Accuracy: 0.341

The poor performance for age is due to a highly imbalanced dataset. The data was not imbalanced with respect to gender and performance in detection and gender classification proves that the model works as expected.

The accuracy calculations include the cases where the model is not able to detect the faces properly. Since it doesn't detect the face, it fails to classify them.

# Model Comparison
For comparison, I also trained independent models for classification given a face. These models had the following performance:

- Gender classification accuracy: 0.834
- Age classification accuracy: 0.364

Please note that these models already had correct faces as input compared to my single model which had to detect faces first. So overall the performance of the single model is actually comparable to these.

All code is in this repo, and you can read more about it [here][comparison].

# More Details
More detailed overview of this project is on my blog post [here][blog].

[yolo]: https://arxiv.org/abs/1506.02640
[comparison]:  http://www.cylopsis.com/post/computer-vision/face-attribute-detection/#model-comparison
[preprocess]: http://www.cylopsis.com/post/computer-vision/face-attribute-detection/#data-pre-processing
[blog]: http://www.cylopsis.com/post/computer-vision/face-attribute-detection
[non-max]: https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH