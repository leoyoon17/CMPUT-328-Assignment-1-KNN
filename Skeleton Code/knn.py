import tensorflow as tf
import numpy as np


def knn(x_train, y_train, x_test):
    """
    x_train: 60000 x 784 matrix: each row is a flattened image of an MNIST digit
    y_train: 60000 vector: label for x_train
    x_test: 5000 x 784 testing images
    return: predicted y_test which is a 5000 vector
    """
    raise NotImplementedError
