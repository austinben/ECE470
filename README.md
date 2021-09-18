# ECE470 - Detecting Brain Tumors With Machine Learning and MRI Scans

ECE470 - Artificial Intelligence - Summer 2021

 University of Victoria

Ben Austin / Derrick Ushko / David Bishop

## Overview
The objective of this project was to develop and implement a machine learning model that can detect brain tumors in MRI scans. A dataset from Kaggle with over 1000 MRI images was used to train and test a model. The images were categorized as either “yes” or “no” to indicate the presence of tumors. A convolutional neural network was selected as the method for developing a binary classifier. The network was composed of three 2D convolutional layers, three max pooling layers, two dense layers, as well as batch normalization and dropout layers. Input samples were augmented by rescaling, flipping, and rotating the images. Several strategies were used to avoid overfitting the model. The final version of the model achieved an accuracy of 65.9%. The model was compared to similar models that have been published online. The project provided an excellent opportunity to study applications of artificial intelligence through the development of a machine learning model to detect brain tumors in MRI scans.

## Requirements

First, install needed requirements, preferably in a virtual environment, by using the following:

`pip install -r requirements.txt`

## Running

### Start a New Model

`python3 model.py new`

### Run Model with Previously Trained Weights

`python3 model.py`

### Fit Model with Previously Trained Weights

`python3 model.py fit`

### Display Intermediate Activations

`python3 model.py display`
