# Audio Classification using Mel Spectrograms

This repository contains code for a simple neural network model that performs audio classification using Mel spectrograms. The model is trained to classify audio samples into two classes. The code includes data preprocessing, model definition, training, and evaluation on a test dataset.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Librosa
- Matplotlib
- Scikit-learn

You can install the required Python packages using the following command:
(Ubuntu/MacOS X)

bash 
pip install -r requirements.txt
Getting Started

Clone the repository:
bash
git clone https://github.com/your-username/audio-classification.git
cd audio-classification
Mount Google Drive if using Colab:
python

from google.colab import drive
drive.mount("/content/drive")
Install necessary packages and extract the dataset:
bash

!pip install unrar
!unrar x /content/drive/MyDrive/data.rar
Data Preparation

The dataset is assumed to be organized in folders, where each folder represents a class. Adjust the code in load_data.py to match your specific dataset structure.
The script dynamically generates labels based on folder names.
Model Architecture

The neural network model consists of convolutional layers, max-pooling, flattening, dense layers, and dropout.
The model is defined in train_model.py.
Training

The model is trained using the training set and validated on a validation set.
Adjust hyperparameters, model architecture, and training parameters in the script.
Evaluation

The trained model is evaluated on a test dataset using Mel spectrograms.
The predicted classes are printed to the console.
Results

Check the training log to see the model's performance over epochs.
Consider experimenting with different architectures and hyperparameters to improve results.
Usage

Train the model:
bash

python train_model.py
Evaluate the model on test data:
bash

python evaluate_model.py

Contributing

If you'd like to contribute, please fork the repository, create a new branch, and submit a pull request. Feel free to open issues for any bugs or feature requests.
