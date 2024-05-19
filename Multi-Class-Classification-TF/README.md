## Overview

This project demonstrates the use of transfer learning with ResNet-50 to classify images of flowers. Transfer learning allows us to leverage pre-trained models to improve the accuracy and efficiency of our own models.

## Dataset

The dataset used in this project consists of images of various types of flowers. The sample image shown above is an example of a daisy flower. The dataset is split into training, validation, and testing sets.

## Model

We use ResNet-50, a powerful convolutional neural network pre-trained on the ImageNet dataset. By fine-tuning this model on our flower dataset, we can achieve high accuracy in flower classification with reduced training time.

## Files

- `Transfer_Learning(Restnet50).ipynb`: This Jupyter notebook contains the implementation of transfer learning using ResNet-50. It includes data preprocessing, model building, training, and evaluation steps.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/ajafarsadiq2002/flower-classification.git
   ```
2.Install the required packages
3.Run the code


## Usage
Ensure you have the dataset in the appropriate directory structure.
Run the notebook cell by cell to train and evaluate the model.
Use the trained model to classify new flower images.

## Results
The model achieves an accuracy of 99% on the validation set and 91% on the test set


## Conclusion
Transfer learning with ResNet-50 provides a powerful tool for image classification tasks, achieving high accuracy with minimal training time. This project can be extended to other types of image datasets for further experimentation.
