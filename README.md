Deep Learning Model Comparisons on Image Datasets
Project Overview
This project involves the implementation and comparison of three popular deep learning architectures: VGG, ResNet, and GoogLeNet. Each model is trained and evaluated on three different image datasets: MNIST, CIFAR-10, and CIFAR-100. The project aims to analyze the performance of these models in terms of accuracy, precision, recall, and F1 score.

Repository Contents
cw3.py: The main Python script with model implementations and experiments.
logs/: Directory to store training logs for each model and dataset.
classified_images/: Directory containing output images showing model predictions.

Requirements
TensorFlow
Keras
NumPy
Matplotlib
scikit-learn
OpenCV
PIL
Installation
To install the necessary libraries, create a virtual environment and run:


pip install -r requirements.txt
Models Implemented
VGG: A deep CNN known for its simplicity and depth.
ResNet: A residual network known for its "skip connections."
GoogLeNet (Inception): A network known for its inception modules.
Datasets
MNIST: Handwritten digit recognition.
CIFAR-10: Object recognition with 10 classes.
CIFAR-100: Object recognition with 100 classes.

Usage
To run the experiments across all models and datasets, execute:


python cw3.py
Output
The script will output the accuracy, precision, recall, and F1 scores for each model on each dataset.
Classified images and training logs will be saved in their respective directories.
Contributions
Contributions, issues, and feature requests are welcome. Feel free to check issues page if you want to contribute.
