import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

"""
A single hidden layer binary classifier.
Technically don't need this and dnn_model, but this is
a good way to introduce things.
"""
def baby_model(input_shape,
                num_hidden_layer_nodes = 32):

    X_input = Input(input_shape)
    X = Dense(num_hidden_layer_nodes, activation="relu")(X_input)
    X = Dense(1, activation="sigmoid")(X)
    model = Model(inputs=X_input, outputs=X)

    return model

"""
Multiple hidden layers in the network and possibility of multiple categories.
"""
def dnn_model(input_shape,
            num_classes,
            hidden_layers=[32, 32]):
    X_input = Input(input_shape)

    X = X_input
    for num_nodes in hidden_layers:
        X = Dense(num_nodes, activation="relu")(X)
    X = Dense(num_classes, activation="relu")(X)

    model = Model(inputs=X_input, outputs=X)