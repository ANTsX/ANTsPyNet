
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Activation, Flatten,
                          Conv2D, MaxPooling2D,
                          Conv3D, MaxPooling3D)

from ..utilities import (SpatialTransformer2D, SpatialTransformer3D)

import numpy as np
import tensorflow as tf

def create_simple_classification_with_spatial_transformer_network_model_2d(input_image_size,
                                                                           resampled_size=(30, 30),
                                                                           number_of_classification_labels=10):
    """
    2-D implementation of the spatial transformer network.

    Creates a keras model of the spatial transformer network:

            https://arxiv.org/abs/1506.02025

    based on the following python Keras model:

            https://github.com/oarriaga/STN.keras/blob/master/src/models/STN.py

    @param inputImageSize Used for specifying the input tensor shape.  The
    shape (or dimension) of that tensor is the image dimensions followed by
    the number of channels (e.g., red, green, and blue).  The batch size
    (i.e., number of training images) is not specified a priori.
    @param resampledSize resampled size of the transformed input images.
    @param numberOfClassificationLabels Number of classes.

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    resampled_size : tuple of length 2
        Resampled size of the transformed input images.

    number_of_classification_labels : integer
        Number of units in the final dense layer.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_simple_classification_with_spatial_transformer_network_model_2d((128, 128, 1))
    >>> model.summary()
    """

    def get_initial_weights_2d(output_size):
        b = np.zeros((2, 3), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1

        W = np.zeros((output_size, 6), dtype='float32')

        weights = [W, b.flatten()]
        return(weights)

    inputs = Input(shape = input_image_size)

    localization = inputs
    localization = MaxPooling2D(pool_size=(2, 2))(localization)
    localization = Conv2D(filters=20,
                          kernel_size=(5, 5))(localization)
    localization = MaxPooling2D(pool_size=(2, 2))(localization)
    localization = Conv2D(filters=20,
                          kernel_size=(5, 5))(localization)

    localization = Flatten()(localization)
    localization = Dense(units=50)(localization)
    localization = Activation('relu')(localization)

    weights = get_initial_weights_2d(output_size=50)
    localization = Dense(6, kernel_initializer = tf.constant_initializer(weights[0]),
                            bias_initializer = tf.constant_initializer(weights[1]))(localization)

    outputs = SpatialTransformer2D(resampled_size=resampled_size,
                                   transform_type="affine",
                                   interpolator_type="linear")([inputs, localization])
    outputs = Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same')(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Conv2D(filters=32,
                     kernel_size=(3, 3))(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPooling2D(pool_size=(2, 2))(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(units=256)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dense(units=number_of_classification_labels)(outputs)

    outputs = Activation('softmax')(outputs)

    stnModel = Model(inputs=inputs, outputs=outputs)

    return(stnModel)

def create_simple_classification_with_spatial_transformer_network_model_3d(input_image_size,
                                                                           resampled_size=(30, 30, 30),
                                                                           number_of_classification_labels=10):
    """
    3-D implementation of the spatial transformer network.

    Creates a keras model of the spatial transformer network:

            https://arxiv.org/abs/1506.02025

    based on the following python Keras model:

            https://github.com/oarriaga/STN.keras/blob/master/src/models/STN.py

    @param inputImageSize Used for specifying the input tensor shape.  The
    shape (or dimension) of that tensor is the image dimensions followed by
    the number of channels (e.g., red, green, and blue).  The batch size
    (i.e., number of training images) is not specified a priori.
    @param resampledSize resampled size of the transformed input images.
    @param numberOfClassificationLabels Number of classes.

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    resampled_size : tuple of length 3
        Resampled size of the transformed input images.

    number_of_classification_labels : integer
        Number of units in the final dense layer.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_simple_classification_with_spatial_transformer_network_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    def get_initial_weights_3d(output_size):
        b = np.zeros((3, 4), dtype='float32')
        b[0, 0] = 1
        b[1, 1] = 1
        b[2, 2] = 1

        W = np.zeros((output_size, 12), dtype='float32')

        weights = [W, b.flatten()]
        return(weights)

    inputs = Input(shape = input_image_size)

    localization = inputs
    localization = MaxPooling3D(pool_size=(2, 2, 2))(localization)
    localization = Conv3D(filters=20,
                          kernel_size=(5, 5, 5))(localization)
    localization = MaxPooling3D(pool_size=(2, 2, 2))(localization)
    localization = Conv3D(filters=20,
                          kernel_size=(5, 5, 5))(localization)

    localization = Flatten()(localization)
    localization = Dense(units=50)(localization)
    localization = Activation('relu')(localization)

    weights = get_initial_weights_3d(output_size=50)
    localization = Dense(6, kernel_initializer = tf.constant_initializer(weights[0]),
                            bias_initializer = tf.constant_initializer(weights[1]))(localization)

    outputs = SpatialTransformer3D(resampled_size=resampled_size,
                                   transform_type="affine",
                                   interpolator_type="linear")([inputs, localization])
    outputs = Conv3D(filters=32,
                     kernel_size=(3, 3, 3),
                     padding='same')(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)
    outputs = Conv3D(filters=32,
                     kernel_size=(3, 3, 3))(outputs)
    outputs = Activation('relu')(outputs)
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(units=256)(outputs)
    outputs = Activation('relu')(outputs)
    outputs = Dense(units=number_of_classification_labels)(outputs)

    outputs = Activation('softmax')(outputs)

    stnModel = Model(inputs=inputs, outputs=outputs)

    return(stnModel)
