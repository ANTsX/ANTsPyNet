
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Average, Add,
                          Conv2D, Conv2DTranspose,
                          MaxPooling2D, UpSampling2D,
                          Conv3D, Conv3DTranspose,
                          MaxPooling3D, UpSampling3D)

import numpy as np

def create_deep_denoise_super_resolution_model_2d(input_image_size,
                                                  layers=2,
                                                  lowest_resolution=64,
                                                  convolution_kernel_size=(3, 3),
                                                  pool_size=(2, 2),
                                                  strides=(2, 2)
                                                 ):
    """
    2-D implementation of the denoising autoencoder image super resolution deep learning architecture.

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    layers : integer
        Number of architecture layers.

    lowest_resolution : integer
        Number of filters at the beginning and end of the architecture.

    convolution_kernel_size : 2-d tuple
        specifies the kernel size during the encoding path.

    pool_size : 2-d tuple
         Defines the region for each pooling layer.

    strides : 2-d tuple
         Defines the stride length in each direction.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_deep_denoise_super_resolution_model_2d((128, 128, 1))
    >>> model.summary()
    """

    inputs = Input(shape = input_image_size)

    # encoding layers

    pool = None
    conv = None

    encoding_convolution_layers = []
    for i in range(layers):
        number_of_filters = lowest_resolution * 2 ** i
        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(inputs)
        else:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(pool)

        layer = Conv2D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='relu',
                       padding='same')(conv)
        encoding_convolution_layers.append(layer)

        pool = MaxPooling2D(pool_size=pool_size,
                            strides=strides,
                            padding='same')(encoding_convolution_layers[i])

    number_of_filters = lowest_resolution * 2 ** layers

    outputs = Conv2D(filters=number_of_filters,
                     kernel_size=convolution_kernel_size,
                     activation='relu',
                     padding='same')(pool)

    # upsampling layers

    for i in range(layers):
        number_of_filters = lowest_resolution * 2 ** (layers - i - 1)

        outputs = UpSampling2D()(outputs)

        conv = Conv2D(filters=number_of_filters,
                      kernel_size=convolution_kernel_size,
                      activation='relu',
                      padding='same')(outputs)
        conv = Conv2D(filters=number_of_filters,
                      kernel_size=convolution_kernel_size,
                      activation='relu',
                      padding='same')(conv)

        outputs = Add()([encoding_convolution_layers[layers - i -1], conv])
        outputs = UpSampling2D()(outputs)

    number_of_channels = input_image_size[-1]

    final_convolution_kernel_size = tuple(np.array(convolution_kernel_size) + 2)

    outputs = Conv2D(filters=number_of_channels,
                     kernel_size=final_convolution_kernel_size,
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)


def create_deep_denoise_super_resolution_model_3d(input_image_size,
                                                  layers=2,
                                                  lowest_resolution=64,
                                                  convolution_kernel_size=(3, 3, 3),
                                                  pool_size=(2, 2, 2),
                                                  strides=(2, 2, 2)
                                                 ):
    """
    3-D implementation of the denoising autoencoder image super resolution deep learning architecture.

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    layers : integer
        Number of architecture layers.

    lowest_resolution : integer
        Number of filters at the beginning and end of the architecture.

    convolution_kernel_size : 3-d tuple
        specifies the kernel size during the encoding path.

    pool_size : 3-d tuple
         Defines the region for each pooling layer.

    strides : 3-d tuple
         Defines the stride length in each direction.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_deep_denoise_super_resolution_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    inputs = Input(shape = input_image_size)

    # encoding layers

    pool = None
    conv = None

    encoding_convolution_layers = []
    for i in range(layers):
        number_of_filters = lowest_resolution * 2 ** i
        if i == 0:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(inputs)
        else:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same')(pool)

        layer = Conv3D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='relu',
                       padding='same')(conv)
        encoding_convolution_layers.append(layer)

        pool = MaxPooling3D(pool_size=pool_size,
                            strides=strides,
                            padding='same')(encoding_convolution_layers[i])

    number_of_filters = lowest_resolution * 2 ** layers

    outputs = Conv3D(filters=number_of_filters,
                     kernel_size=convolution_kernel_size,
                     activation='relu',
                     padding='same')(pool)

    # upsampling layers

    for i in range(layers):
        number_of_filters = lowest_resolution * 2 ** (layers - i - 1)

        outputs = UpSampling3D()(outputs)

        conv = Conv3D(filters=number_of_filters,
                      kernel_size=convolution_kernel_size,
                      activation='relu',
                      padding='same')(outputs)
        conv = Conv3D(filters=number_of_filters,
                      kernel_size=convolution_kernel_size,
                      activation='relu',
                      padding='same')(conv)

        outputs = Add()([encoding_convolution_layers[layers - i -1], conv])
        outputs = UpSampling3D()(outputs)

    number_of_channels = input_image_size[-1]

    final_convolution_kernel_size = tuple(np.array(convolution_kernel_size) + 2)

    outputs = Conv3D(filters=number_of_channels,
                     kernel_size=final_convolution_kernel_size,
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)

