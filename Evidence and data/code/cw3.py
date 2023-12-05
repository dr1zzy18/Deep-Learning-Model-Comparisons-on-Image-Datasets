#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, Flatten,
                                     Dropout, Add, Activation, BatchNormalization,
                                     GlobalAveragePooling2D, Concatenate, AveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar100
import matplotlib.pyplot as plt
import datetime
import os
import cv2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import gc
import tensorflow.keras as keras
from PIL import Image
import math
import os
import datetime
from sklearn.metrics import confusion_matrix


def build_vgg(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add more Conv2D and MaxPooling2D layers as needed

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def build_resnet(input_shape, num_classes):
    def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), use_conv_shortcut=False):
        shortcut = x
        if use_conv_shortcut:
            shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides,
                              kernel_regularizer=l2(1e-4))(x)
            shortcut = BatchNormalization()(shortcut)

        x = Conv2D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)

        x = Add()([shortcut, x])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, strides=(2, 2), use_conv_shortcut=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, strides=(2, 2), use_conv_shortcut=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, strides=(2, 2), use_conv_shortcut=True)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def build_googlenet(input_shape, num_classes):
    def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

        conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)

        conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)

        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool)

        output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
        return output

    input_layer = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,
                         filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)

    x = inception_module(x, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192,
                         filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)

    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    # Add more inception modules as needed

    x = AveragePooling2D((2, 2), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model






def preprocess_data(x_train, x_test, target_shape=(32, 32, 3)):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    x_train_resized = []
    x_test_resized = []

    for img in x_train:
        img_resized = cv2.resize(img, target_shape[:2])
        img_resized = np.stack([img_resized] * 3, axis=-1)  # Convert to 3 channels
        x_train_resized.append(img_resized)

    for img in x_test:
        img_resized = cv2.resize(img, target_shape[:2])
        img_resized = np.stack([img_resized] * 3, axis=-1)  # Convert to 3 channels
        x_test_resized.append(img_resized)

    x_train = np.array(x_train_resized)
    x_test = np.array(x_test_resized)

    return x_train, x_test


def load_and_preprocess_extra_dataset():
    (x_train_extra, y_train_extra), (x_test_extra, y_test_extra) = cifar10.load_data()

    x_train_extra = x_train_extra.astype('float32') / 255
    x_test_extra = x_test_extra.astype('float32') / 255

    y_train_extra = to_categorical(y_train_extra, 10)
    y_test_extra = to_categorical(y_test_extra, 10)

    return x_train_extra, y_train_extra, x_test_extra, y_test_extra

def train_model(model, data_gen, x_train, y_train, x_test, y_test, dataset_name, model_name, epochs=100, batch_size=64):
    start_time = datetime.datetime.now()
    print(f"Model: {model_name}, Dataset: {dataset_name}, Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    history = model.fit(data_gen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)

    end_time = datetime.datetime.now()
    print(f"Model: {model_name}, Dataset: {dataset_name}, Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

    # Print the test accuracy along with model name and dataset name
    print(f"Model: {model_name}, Dataset: {dataset_name}, Test Accuracy: {test_accuracy:.4f}")

    return model, history, start_time, end_time

def save_logs(model_name, history, start_time, end_time):
    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "history.txt"), "w") as f:
        f.write(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Training History:\n")
        f.write(str(history))


def save_classified_images(model, x_test, y_test, dataset_name, network_name, num_images=10):
    # Get predictions
    y_pred = model.predict(x_test)

    # Get the indices of the highest probability predictions
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Create a directory to save the images if it does not exist
    base_dir = 'classified_images'
    dataset_dir = os.path.join(base_dir, dataset_name)
    network_dir = os.path.join(dataset_dir, network_name)

    if not os.path.exists(network_dir):
        os.makedirs(network_dir)

    # Set up grid dimensions
    grid_dim = int(math.sqrt(num_images))
    if grid_dim * grid_dim < num_images:
        grid_dim += 1

    # Create a grid of images
    fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(15, 15))

    for i in range(grid_dim * grid_dim):
        ax = axes[i // grid_dim, i % grid_dim]
        if i < num_images:
            ax.imshow(x_test[i], cmap='gray')
            ax.set_title(f"True: {y_true_labels[i]} - Pred: {y_pred_labels[i]}")
        ax.axis('off')

    # Save the grid as a single image
    plt.savefig(os.path.join(network_dir, f'classified_images_grid.png'), dpi=300)

    # Close the plot
    plt.close()




def visualize_results(vgg_history, resnet_history, googlenet_history):
    plt.figure(figsize=(10, 6))
    plt.plot(vgg_history.history['val_accuracy'], label='VGG')
    plt.plot(resnet_history.history['val_accuracy'], label='ResNet')
    plt.plot(googlenet_history.history['val_accuracy'], label='GoogleNet')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()





def experiment_mnist():
    # Load and preprocess MNIST data
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    x_train_mnist, x_test_mnist = preprocess_data(x_train_mnist, x_test_mnist)
    data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    y_train_mnist = to_categorical(y_train_mnist, 10)
    y_test_mnist = to_categorical(y_test_mnist, 10)

    # Train and evaluate models on MNIST
    vgg_model = build_vgg(input_shape=(32, 32, 3), num_classes=10)
    resnet_model = build_resnet(input_shape=(32, 32, 3), num_classes=10)
    googlenet_model = build_googlenet(input_shape=(32, 32, 3), num_classes=10)

    vgg_model,vgg_history,vgg_start_time, vgg_end_time = train_model(vgg_model, data_gen, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,"mnist","vgg")
    resnet_model, resnet_history,resnet_start_time, resnet_end_time = train_model(resnet_model, data_gen, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,"mnist","resnet")
    googlenet_model, googlenet_history,googlenet_start_time, googlenet_end_time = train_model(googlenet_model, data_gen, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,"mnist","googlenet")

    save_classified_images(vgg_model, x_test_mnist, y_test_mnist, "mnist", "vgg")
    save_classified_images(resnet_model, x_test_mnist, y_test_mnist, "mnist", "resnet")
    save_classified_images(googlenet_model, x_test_mnist, y_test_mnist, "mnist", "googlenet")

    # Save logs and visualize results
    save_logs("mnist_vgg_logs.txt", vgg_history.history,vgg_start_time, vgg_end_time)
    save_logs("mnist_resnet_logs.txt", resnet_history.history,resnet_start_time, resnet_end_time)
    save_logs("mnist_googlenet_logs.txt", googlenet_history.history, googlenet_start_time, googlenet_end_time)

    visualize_results(vgg_history, resnet_history, googlenet_history)
    return vgg_model, resnet_model, googlenet_model



def experiment_cifar():
    # Load and preprocess CIFAR-10 data
    x_train_extra, y_train_extra, x_test_extra, y_test_extra = load_and_preprocess_extra_dataset()
    data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

    # Train and evaluate models on CIFAR-10
    vgg_model = build_vgg(input_shape=(32, 32, 3), num_classes=10)
    resnet_model = build_resnet(input_shape=(32, 32, 3), num_classes=10)
    googlenet_model = build_googlenet(input_shape=(32, 32, 3), num_classes=10)

    vgg_model,vgg_history,vgg_start_time, vgg_end_time = train_model(vgg_model, data_gen, x_train_extra, y_train_extra, x_test_extra, y_test_extra,"cifar-10","vgg")
    resnet_model, resnet_history,resnet_start_time, resnet_end_time = train_model(resnet_model, data_gen, x_train_extra, y_train_extra, x_test_extra, y_test_extra,"cifar-10","resnet")
    googlenet_model, googlenet_history,googlenet_start_time, googlenet_end_time = train_model(googlenet_model, data_gen, x_train_extra, y_train_extra, x_test_extra, y_test_extra,"cifar-10","googlenet")

    save_classified_images(vgg_model, x_test_extra, y_test_extra, "cifar10", "vgg")
    save_classified_images(resnet_model, x_test_extra, y_test_extra, "cifar10", "resnet")
    save_classified_images(googlenet_model, x_test_extra, y_test_extra, "cifar10", "googlenet")


    # Save logs and visualize results
    save_logs("cifar_vgg_logs.txt", vgg_history.history,vgg_start_time, vgg_end_time)
    save_logs("cifar_resnet_logs.txt", resnet_history.history,resnet_start_time, resnet_end_time)
    save_logs("cifar_googlenet_logs.txt", googlenet_history.history, googlenet_start_time, googlenet_end_time)

    visualize_results(vgg_history, resnet_history, googlenet_history)
    return vgg_model, resnet_model, googlenet_model


def experiment_cifar100():
    # Load and preprocess CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    # Define models
    vgg_model = build_vgg(input_shape=(32, 32, 3), num_classes=100)
    resnet_model = build_resnet(input_shape=(32, 32, 3), num_classes=100)
    googlenet_model = build_googlenet(input_shape=(32, 32, 3), num_classes=100)

    # Train and evaluate models
    vgg_model, vgg_history,vgg_start_time, vgg_end_time = train_model(vgg_model, data_gen, x_train, y_train, x_test, y_test,"cifar-100","vgg")
    resnet_model, resnet_history,resnet_start_time, resnet_end_time = train_model(resnet_model, data_gen, x_train, y_train, x_test, y_test,"cifar-100","resnet")
    googlenet_model, googlenet_history,googlenet_start_time, googlenet_end_time = train_model(googlenet_model, data_gen, x_train, y_train, x_test, y_test,"cifar-100","googlenet")

    save_classified_images(vgg_model, x_test, y_test, "cifar100", "vgg")
    save_classified_images(resnet_model, x_test, y_test, "cifar100", "resnet")
    save_classified_images(googlenet_model, x_test, y_test, "cifar100", "googlenet")

    # Save logs and visualize results
    save_logs("cifar100_vgg_logs.txt", vgg_history.history,vgg_start_time, vgg_end_time)
    save_logs("cifar100_resnet_logs.txt", resnet_history.history,resnet_start_time, resnet_end_time)
    save_logs("cifar100_googlenet_logs.txt", googlenet_history.history, googlenet_start_time, googlenet_end_time)

    visualize_results(vgg_history, resnet_history, googlenet_history)
    return vgg_model, resnet_model, googlenet_model




if __name__ == "__main__":
    vgg_model_mnist, resnet_model_mnist, googlenet_model_mnist = experiment_mnist()
    # Clear memory
    keras.backend.clear_session()
    gc.collect()

    vgg_model_cifar10, resnet_model_cifar10, googlenet_model_cifar10 = experiment_cifar()
    # Clear memory
    keras.backend.clear_session()
    gc.collect()

    vgg_model_cifar100, resnet_model_cifar100, googlenet_model_cifar100 = experiment_cifar100()
    # Clear memory
    tf.keras.backend.clear_session()
    gc.collect()



# Load and preprocess MNIST data
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_mnist, x_test_mnist = preprocess_data(x_train_mnist, x_test_mnist)
data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
y_train_mnist = to_categorical(y_train_mnist, 10)
y_test_mnist = to_categorical(y_test_mnist, 10)


# Load and preprocess CIFAR-10 data
x_train_extra, y_train_extra, x_test_extra, y_test_extra = load_and_preprocess_extra_dataset()
data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Load and preprocess CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)


def calculate_scores(model_name):

    import numpy as np

    if model_name == "vgg":
        # Predict the labels for MNIST
        y_pred_mnist = vgg_model_mnist.predict(x_test_mnist)
        y_pred_mnist = np.argmax(y_pred_mnist, axis=1)

        # Predict the labels for CIFAR-10
        y_pred_cifar10 = vgg_model_cifar10.predict(x_test_extra)
        y_pred_cifar10 = np.argmax(y_pred_cifar10, axis=1)

        # Predict the labels for CIFAR-100
        y_pred_cifar100 = vgg_model_cifar100.predict(x_test)
        y_pred_cifar100 = np.argmax(y_pred_cifar100, axis=1)

    if model_name == "resnet":

        # Predict the labels for MNIST
        y_pred_mnist = resnet_model_mnist.predict(x_test_mnist)
        y_pred_mnist = np.argmax(y_pred_mnist, axis=1)

        # Predict the labels for CIFAR-10
        y_pred_cifar10 = resnet_model_cifar10.predict(x_test_extra)
        y_pred_cifar10 = np.argmax(y_pred_cifar10, axis=1)

        # Predict the labels for CIFAR-100
        y_pred_cifar100 = resnet_model_cifar100.predict(x_test)
        y_pred_cifar100 = np.argmax(y_pred_cifar100, axis=1)

    if model_name == "googlenet":

        # Predict the labels for MNIST
        y_pred_mnist = googlenet_model_mnist.predict(x_test_mnist)
        y_pred_mnist = np.argmax(y_pred_mnist, axis=1)

        # Predict the labels for CIFAR-10
        y_pred_cifar10 = googlenet_model_cifar10.predict(x_test_extra)
        y_pred_cifar10 = np.argmax(y_pred_cifar10, axis=1)

        # Predict the labels for CIFAR-100
        y_pred_cifar100 = googlenet_model_cifar100.predict(x_test)
        y_pred_cifar100 = np.argmax(y_pred_cifar100, axis=1)



    y_true_mnist = y_test_mnist
    y_true_cifar10 = y_test_extra
    y_true_cifar100 = y_test

    def confusion_matrix_elements_mnist(y_true, y_pred):
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoded labels back to integers

        cm = confusion_matrix(y_true, y_pred)

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        return TP, FP, FN, TN


    def confusion_matrix_elements_cifar(y_true, y_pred):
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)  # Convert one-hot encoded labels back to integers

        cm = confusion_matrix(y_true, y_pred)

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        return TP, FP, FN, TN

    TP0, FP0, FN0, TN0 = confusion_matrix_elements_mnist(y_true_mnist, y_pred_mnist)
    TP1, FP1, FN1, TN1 = confusion_matrix_elements_cifar(y_true_cifar10, y_pred_cifar10)
    TP2, FP2, FN2, TN2 = confusion_matrix_elements_cifar(y_true_cifar100, y_pred_cifar100)

    def precision(TP, FP):
        return TP / (TP + FP) if TP + FP > 0 else 0

    def recall(TP, FN):
        return TP / (TP + FN) if TP + FN > 0 else 0

    def f1_score(precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


    avg_precision_mnist = np.mean([precision(tp, fp) for tp, fp in zip(TP0, FP0)])
    avg_recall_mnist = np.mean([recall(tp, fn) for tp, fn in zip(TP0, FN0)])
    avg_f1_score_mnist = np.mean([f1_score(precision(tp, fp), recall(tp, fn)) for tp, fp, fn in zip(TP0, FP0, FN0)])

    print("Average Precision for MNIST:", avg_precision_mnist)
    print("Average Recall for MNIST:", avg_recall_mnist)
    print("Average F1-score for MNIST:", avg_f1_score_mnist)

    avg_precision_cifar10 = np.mean([precision(tp, fp) for tp, fp in zip(TP1, FP1)])
    avg_recall_cifar10 = np.mean([recall(tp, fn) for tp, fn in zip(TP1, FN1)])
    avg_f1_score_cifar10 = np.mean([f1_score(precision(tp, fp), recall(tp, fn)) for tp, fp, fn in zip(TP1, FP1, FN1)])

    print("Average Precision for CIFAR-10:", avg_precision_cifar10)
    print("Average Recall for CIFAR-10:", avg_recall_cifar10)
    print("Average F1-score for CIFAR-10:", avg_f1_score_cifar10)

    avg_precision_cifar100 = np.mean([precision(tp, fp) for tp, fp in zip(TP2, FP2)])
    avg_recall_cifar100 = np.mean([recall(tp, fn) for tp, fn in zip(TP2, FN2)])
    avg_f1_score_cifar100 = np.mean([f1_score(precision(tp, fp), recall(tp, fn)) for tp, fp, fn in zip(TP2, FP2, FN2)])

    print("Average Precision for CIFAR-100:", avg_precision_cifar100)
    print("Average Recall for CIFAR-100:", avg_recall_cifar100)
    print("Average F1-score for CIFAR-100:", avg_f1_score_cifar100)




calculate_scores("vgg")

calculate_scores("resnet")

calculate_scores("googlenet")
