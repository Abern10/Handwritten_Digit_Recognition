# data_preprocessing.py

import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_preprocess_data():
    # Load EMNIST dataset
    emnist_train, emnist_test = tfds.load('emnist', split=['train', 'test'])

    # Define a preprocessing function
    def preprocess(data):
        image = tf.cast(data['image'], tf.float32) / 255.0  # Normalize pixel values
        label = data['label']
        return image, label

    # Apply preprocessing to train and test data
    emnist_train = emnist_train.map(preprocess)
    emnist_test = emnist_test.map(preprocess)

    # Extract numpy arrays from datasets
    x_train, y_train = [], []
    for image, label in tfds.as_numpy(emnist_train):
        x_train.append(image)
        y_train.append(label)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test, y_test = [], []
    for image, label in tfds.as_numpy(emnist_test):
        x_test.append(image)
        y_test.append(label)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Split the data into training, validation, and testing sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, y_train, x_val, y_val, x_test, y_test