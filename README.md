

# An Explainable Artificial Intelligence approach to Multimodal Skin Lesion Classification

## Motivation
Motivation
With the introduction of deep learning algorithms, artificial intelligence has recently shown
extraordinary performance for a range of tasks, from computer vision to natural language processing.
They have infringed on many various fields and disciplines as the research progresses. One of them is
the healthcare sector, which requires a high level of transparency along with accountability. Health
professionals deal with multiple sources of data in their daily life. This research aims to improve the
effectiveness of skin lesion classification for cancer early detection by utilizing different modalities in
the form of patient metadata and skin lesion images.

## Salient contributions
The research presents the following salient contributions:

1. A novel unimodal technique based on Random Forest Classifier and Multilayer Perceptron
has been developed to perform skin lesion classification using patient metadata. We proposed
unimodal networks based on Convolutional Neural Networks, pre-trained models such as
ResNet50, IncepV3, XcepNet, and VGG19 used as feature extractors to perform skin lesion
classification using skin lesion images of the patients respectively.

2. Multimodal frameworks have been proposed based on Convolutional Neural Networks and
pre-trained models (ResNet50, IncepV3, XcepNet, and VGG19) using a late fusion approach.

3. A scientific method for comparing and contrasting unimodal and multimodal networks. Our
results unequivocally demonstrate that the proposed custom multimodal Convolutional
Network segmented the skin lesions with a Sensitivity of 89.03% and a Specificity of 89.12%.
The proposed multimodal method outperformed unimodal methods.

## EXPERIMENTAL RESULTS 

#### Accuracy Comparison in binary classification

Model|Training Accuracy (%)|Validation Accuracy(%)|Test Accuracy(%)
-----|---------------------|----------------------|----------------
Unimodal-Random Forest classifier|81.34|86.00|81.00
Unimodal-ResNet50|80.88|92.32|84.65
Unimodal-XceptionNet|65.89|79.22|85.13
Unimodal-IncepV3|66.28|73.18|79.28
Unimodal-VGG19|69.25|84.95|81.02
Unimodal-CNN|81.52|89.54|85.25
Multimodal-ResNet50|87.29|93.35|84.93
Multimodal-XceptionNet|69.80|82.29|86.82
Multimodal-IncepV3|72.76|84.34|85.84
Multimodal-VGG19|81.61|89.97|83.90
Multimodal-CNN|82.07|92.84|89.01

#### Performance Comparison in binary classification

Model|Precision|Recall|F1-Score
-----|---------|------|--------
Unimodal-Random forest classifier|0.7123|0.6850|0.6950
Unimodal-ResNet50|0.8993|0.8450|0.8355
Unimodal-XceptionNet|0.7982|0.8600|0.8513
Unimodal-IncepV3|0.7424|0.8123|0.8966
Unimodal-VGG19|0.8320|0.7773|0.8037
Unimodal-CNN|0.8921|0.8847|0.8845
Multimodal-ResNet50|0.8552|0.8433|0.8420
Multimodal-XceptionNet|0.8789|0.8672|0.8681
Multimodal-IncepV3|0.8614|0.8584|0.8581
Multimodal-VGG19|0.8424|0.8386|0.8390
Multimodal-CNN|0.8923|0.8921|0.8921

Observe that our Multimodal CNN outperforms all the models. Also adding one more modality (patient’s metadata) to the networks increases the classification accuracy. Accuracy is not considered as the only evaluation criteria especially when dealing with imbalanced data. Therefore, precision ,recall and F1-Score values are also taken into consideration.
### MODEL OUTPUT


![GRAD-CAM](https://user-images.githubusercontent.com/51873771/234573456-072e5886-3477-4257-9a5c-59bb33f4c9e8.png)

### Built With
We trained all the networks using Google Colab having a GPU machine with the following
specifications: 
* Hardware: CPU - Intel i7-6700 @ 4.00Ghz, GPU - NVIDIA TITAN X 12Gb, RAM- 32GB DDR5 
* Software: Tensor-flow version 2.12.0 and Python version: 3.9.1

The following frameworks and libraries used throughout this project are provided below. To know more, go through the provided links.

* Tensorflow [https://www.tensorflow.org/]
* Keras [https://keras.io/]
* Pandas [https://pandas.pydata.org]
* Numpy [https://numpy.org/]
* Matplotlib [https://matplotlib.org/]
* LIME [https://lime-ml.readthedocs.io/en/latest/]
* SHAP [https://shap.readthedocs.io/en/latest/index.html]
* GRAD-CAM [https://keras.io/examples/vision/grad_cam/]

## Getting Started

Separate notebooks are created for all experiments. Download HAM10000 data set using the provided links. 

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Kaggle API command for HAM10000 Skin Lesion Dataset is provided below:

[kaggle datasets download -d kmader/skin-cancer-mnist-ham10000]


## Contact

Vanshika Sharma 
* Mail at vanshika170431@dei.ac.in
* Twitter handle: [@vanshika__says](https://twitter.com/@vanshika__says) 



Project Link: [https://github.com/vanshika1501/An-Explainable-Deep-Learning-Approach-for-Multimodal-Skin-Lesion-Classification](https://github.com/vanshika1501/An-Explainable-Deep-Learning-Approach-for-Multimodal-Skin-Lesion-Classification)




