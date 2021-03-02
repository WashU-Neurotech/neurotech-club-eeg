import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

"""
A single hidden layer binary classifier.
Technically don't need this and dnn_model, but this is
a good way to introduce things.
"""
def baby_binary(input_shape,
                num_hidden_layer_nodes = 32,
                kernel_regularizer_type=None,
                kernel_regularizer_amount=None):
    if kernel_regularizer_type == "l2":
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer_amount)
    elif kernel_regularizer_type == "l1":
        kernel_regularizer = tf.keras.regularizers.l1(kernel_regularizer_amount)
    else:
        kernel_regularizer = None

    X_input = Input(input_shape)
    X = Dense(num_hidden_layer_nodes, activation="relu",
        kernel_regularizer=kernel_regularizer)(X_input)
    X = Dense(1, activation="sigmoid", 
        kernel_regularizer=kernel_regularizer)(X)
    
    model = Model(inputs=X_input, outputs=X)

    return model

"""
Multiple hidden layers in the network.
"""
def dnn_binary(input_shape,
            hidden_layers=[32, 32],
            kernel_regularizer_type=None,
            kernel_regularizer_amount=None):
    if kernel_regularizer_type == "l2":
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer_amount)
    elif kernel_regularizer_type == "l1":
        kernel_regularizer = tf.keras.regularizers.l1(kernel_regularizer_amount)
    else:
        kernel_regularizer = None

    X_input = Input(input_shape)

    X = X_input
    for num_nodes in hidden_layers:
        X = Dense(num_nodes, activation="relu",
            kernel_regularizer=kernel_regularizer
            )(X)

        X = tf.keras.layers.Dropout(0.25)(X)
    X = Dense(1, activation="sigmoid",
            kernel_regularizer=kernel_regularizer
        )(X)

    model = Model(inputs=X_input, outputs=X)

    return model