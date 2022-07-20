import tensorflow as tf
from tensorflow import keras
import download
import numpy as np


def create_model():
    """create keras model

    Returns:
        _type_: keras model with a flatten layer and two dense
        layers sequential
    """
    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model):
    x_train = download
