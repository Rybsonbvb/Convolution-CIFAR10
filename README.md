# CIFAR-10 Image Classification with CNN (PyTorch)

## Project Overview

This project implements a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset using PyTorch.

The goal of this project was to build a complete deep learning pipeline including:

* data preprocessing and normalization
* data augmentation
* model training with early stopping
* performance evaluation
* confusion matrix analysis
* single image inference

The repository demonstrates an end-to-end machine learning workflow and production-style project structure.

---

## Dataset

The project uses the CIFAR-10 dataset, which contains 60,000 32×32 RGB images across 10 classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

Dataset split:

* Training: 80%
* Validation: 20%
* Test: official test split

The dataset is downloaded automatically using torchvision.

---

## Model Architecture

Custom convolutional neural network with:

* multiple convolutional blocks
* batch normalization
* ReLU activations
* max pooling
* dropout regularization
* fully connected classifier

### Architecture Summary

Conv → BatchNorm → ReLU blocks
MaxPooling layers
Dropout regularization
Fully connected classifier

The model is designed to balance performance and generalization.

---

## Training Details

* Loss Function: CrossEntropyLoss
* Optimizer: Adam
* Learning Rate: 0.001
* Data Augmentation:

  * random horizontal flip
  * random rotation
* Early stopping based on validation loss

Training and validation metrics are tracked during training.

---

## Results

| Metric                   | Value |
| ------------------------ | ----- |
| Test Accuracy            | 88.10%   |

Model performance is evaluated using:

* classification accuracy
* loss curves
* confusion matrix
* per-class accuracy

---

## Installation

Clone the repository:

git clone https://github.com/Rybsonbvb/Convolution-CIFAR10.git
cd cifar10-image-classification

Install dependencies:

pip install -r requirements.txt

---

## Training

To train the model from scratch:

python run_training.py

The best model weights will be saved in:

models/best_model.pth

---

## Inference (Predict Single Image)

You can predict the class of a single image using:

python predict.py image.png

Output:

* predicted class
* prediction confidence

---

## Project Structure

cifar10-image-classification/

* notebooks/ — exploratory analysis and results visualization
* src/ — model, dataset handling, training, evaluation pipeline
* models/ — saved model weights
* run_training.py — training entry point
* predict.py — inference script
* README.md — project documentation

---

## Reproducibility

The project uses fixed random seeds for reproducibility of results.

---

## Future Improvements

* transfer learning using ResNet
* hyperparameter tuning
* experiment tracking (TensorBoard or Weights & Biases)
* model deployment API
* model comparison experiments

---

## Technologies Used

* PyTorch
* Torchvision
* NumPy
* Matplotlib
* Scikit-learn

---

## Author

Igor Rybiński
