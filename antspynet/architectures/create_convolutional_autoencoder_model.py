
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape

import numpy as np
import math

def create_convolutional_autoencoder_model_2d(input_image_size,
                                              number_of_filters_per_layer=(32, 64, 128, 10),
                                              convolution_kernel_size=(5, 5),
                                              deconvolution_kernel_size=(5, 5)
                                             ):
    """
    Function for creating a 2-D symmetric convolutional autoencoder model.

    Builds an autoencoder based on the specified array definining the
    number of units in the encoding branch.  Ported from the Keras python
    implementation here:

    https://github.com/XifengGuo/DEC-keras

    Arguments
    ---------
    input_image_size : tuple
        A tuple defining the shape of the 2-D input image

    number_of_units_per_layer : tuple
        A tuple defining the number of units in the encoding branch.

    convolution_kernel_size : tuple or scalar
        Kernel size for convolution

    deconvolution_kernel_size : tuple or scalar
        Kernel size for deconvolution

    Returns
    -------
    Keras models
        A convolutional encoder and autoencoder Keras model.

    Example
    -------
    >>> autoencoder, encoder = create_convolutional_autoencoder_model_2d((128, 128, 3))
    >>> autoencoder.summary()
    >>> encoder.summary()
    """

    activation = 'relu'
    strides = (2, 2)

    number_of_encoding_layers = len(number_of_filters_per_layer) - 1

    factor = 2 ** number_of_encoding_layers

    padding = 'valid'
    if input_image_size[0] % factor == 0:
        padding = 'same'

    inputs = Input(shape = input_image_size)

    encoder = inputs

    for i in range(number_of_encoding_layers):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == (number_of_encoding_layers - 1):
            local_padding = padding
            kernel_size = tuple(np.array(convolution_kernel_size) - 2)

        encoder = Conv2D(filters=number_of_filters_per_layer[i],
                         kernel_size=kernel_size,
                         strides=strides,
                         activation=activation,
                         padding=local_padding)(encoder)

    encoder = Flatten()(encoder)
    encoder = Dense(units=number_of_filters_per_layer[-1])(encoder)

    autoencoder = encoder

    penultimate_number_of_filters = \
      number_of_filters_per_layer[number_of_encoding_layers-1]

    input_image_size_factored = ((math.floor(input_image_size[0] / factor)),
                                 (math.floor(input_image_size[1] / factor)))

    number_of_units_for_encoder_output = (penultimate_number_of_filters *
      input_image_size_factored[0] * input_image_size_factored[1])

    autoencoder = Dense(units=number_of_units_for_encoder_output,
                        activation=activation)(autoencoder)
    autoencoder = Reshape(target_shape=(*input_image_size_factored, penultimate_number_of_filters))(autoencoder)

    for i in range(number_of_encoding_layers, 1, -1):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == number_of_encoding_layers:
            local_padding = padding
            kernel_size = tuple(np.array(deconvolution_kernel_size) - 2)

        autoencoder = Conv2DTranspose(filters=number_of_filters_per_layer[i-2],
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=activation,
                                      padding=local_padding)(autoencoder)

    autoencoder = Conv2DTranspose(input_image_size[-1],
                                  kernel_size=deconvolution_kernel_size,
                                  strides=strides,
                                  padding='same')(autoencoder)

    autoencoder_model = Model(inputs=inputs, outputs=autoencoder)
    encoder_model = Model(inputs=inputs, outputs=encoder)

    return([autoencoder_model, encoder_model])


def create_convolutional_autoencoder_model_3d(input_image_size,
                                              number_of_filters_per_layer=(32, 64, 128, 10),
                                              convolution_kernel_size=(5, 5, 5),
                                              deconvolution_kernel_size=(5, 5, 5)
                                             ):
    """
    Function for creating a 3-D symmetric convolutional autoencoder model.

    Builds an autoencoder based on the specified array definining the
    number of units in the encoding branch.  Ported from the Keras python
    implementation here:

    https://github.com/XifengGuo/DEC-keras

    Arguments
    ---------
    input_image_size : tuple
        A tuple defining the shape of the 3-D input image

    number_of_units_per_layer : tuple
        A tuple defining the number of units in the encoding branch.

    convolution_kernel_size : tuple or scalar
        Kernel size for convolution

    deconvolution_kernel_size : tuple or scalar
        Kernel size for deconvolution

    Returns
    -------
    Keras models
        A convolutional encoder and autoencoder Keras model.

    Example
    -------
    >>> autoencoder, encoder = create_convolutional_autoencoder_model_3d((128, 128, 128, 3))
    >>> autoencoder.summary()
    >>> encoder.summary()
    """

    activation = 'relu'
    strides = (2, 2, 2)

    number_of_encoding_layers = len(number_of_filters_per_layer) - 1

    factor = 2 ** number_of_encoding_layers

    padding = 'valid'
    if input_image_size[0] % factor == 0:
        padding = 'same'

    inputs = Input(shape = input_image_size)

    encoder = inputs

    for i in range(number_of_encoding_layers):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == (number_of_encoding_layers - 1):
            local_padding = padding
            kernel_size = tuple(np.array(convolution_kernel_size) - 2)

        encoder = Conv3D(filters=number_of_filters_per_layer[i],
                         kernel_size=kernel_size,
                         strides=strides,
                         activation=activation,
                         padding=local_padding)(encoder)

    encoder = Flatten()(encoder)
    encoder = Dense(units=number_of_filters_per_layer[-1])(encoder)

    autoencoder = encoder

    penultimate_number_of_filters = \
      number_of_filters_per_layer[number_of_encoding_layers-1]

    input_image_size_factored = ((math.floor(input_image_size[0] / factor)),
                                 (math.floor(input_image_size[1] / factor)),
                                 (math.floor(input_image_size[2] / factor)))

    number_of_units_for_encoder_output = (penultimate_number_of_filters *
      input_image_size_factored[0] * input_image_size_factored[1] *
      input_image_size_factored[2])

    autoencoder = Dense(units=number_of_units_for_encoder_output,
                        activation=activation)(autoencoder)
    autoencoder = Reshape(target_shape=(*input_image_size_factored, penultimate_number_of_filters))(autoencoder)

    for i in range(number_of_encoding_layers, 1, -1):
        local_padding = 'same'
        kernel_size = convolution_kernel_size
        if i == number_of_encoding_layers:
            local_padding = padding
            kernel_size = tuple(np.array(deconvolution_kernel_size) - 2)

        autoencoder = Conv3DTranspose(filters=number_of_filters_per_layer[i-2],
                                      kernel_size=kernel_size,
                                      strides=strides,
                                      activation=activation,
                                      padding=local_padding)(autoencoder)

    autoencoder = Conv3DTranspose(input_image_size[-1],
                                  kernel_size=deconvolution_kernel_size,
                                  strides=strides,
                                  padding='same')(autoencoder)

    autoencoder_model = Model(inputs=inputs, outputs=autoencoder)
    encoder_model = Model(inputs=inputs, outputs=encoder)

    return([autoencoder_model, encoder_model])
