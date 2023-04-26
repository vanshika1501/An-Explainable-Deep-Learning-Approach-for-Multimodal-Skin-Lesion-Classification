

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


### Built With
We trained all the networks using Google Colab having a GPU machine with the following
specifications: 
* Hardware: CPU - Intel i7-6700 @ 4.00Ghz, GPU - NVIDIA TITAN X 12Gb, RAM- 32GB DDR5 
* Software: Tensor-flow version 2.12.0 and Python version: 3.9.1

The following frameworks and libraries used throughout this project are provided below. To know more, go through the provided links.

* Tensorflow [https://www.tensorflow.org/]
* Keras [https://keras.io/]
* Pandas [tps://pandas.pydata.org]
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




