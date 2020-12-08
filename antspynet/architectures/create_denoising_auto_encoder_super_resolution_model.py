
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Average, Add,
                          Conv2D, Conv2DTranspose,
                          Conv3D, Conv3DTranspose)

def create_denoising_auto_encoder_super_resolution_model_2d(input_image_size,
                                                            convolution_kernel_sizes=[(3, 3), (5, 5)],
                                                            number_of_encoding_layers=2,
                                                            number_of_filters=64
                                                           ):
    """
    2-D implementation of the denoising autoencoder image super resolution deep learning architecture.

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    convolution_kernel_sizes : list of 2-d tuples
        specifies the kernel size at each convolution layer.  Default values are
        the same as given in the original paper.  The length of kernel size list
        must be 1 greater than the tuple length of the number of filters.

    number_of_encoding_layers : integer
       The number of encoding layers.

    number_of_filters : integer
       The number of filters for each encoding layer.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_denoising_auto_encoder_super_resolution_model_2d((128, 128, 1))
    >>> model.summary()
    """

    inputs = Input(shape = input_image_size)

    outputs = inputs

    encoding_convolution_layers = []
    for i in range(number_of_encoding_layers):
        if i == 0:
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_sizes[0],
                             activation='relu',
                             padding='same')(outputs)
        else:
            layer = Conv2D(filters=number_of_filters,
                           kernel_size=convolution_kernel_sizes[0],
                           activation='relu',
                           padding='same')(outputs)
            encoding_convolution_layers.append(layer)

    outputs = encoding_convolution_layers[-1]

    for i in range(number_of_encoding_layers):
        index = len(encoding_convolution_layers) - i - 1
        deconvolution = Conv2DTranspose(filters=number_of_filters,
                                        kernel_size=convolution_kernel_sizes[0],
                                        padding='same',
                                        activation='relu')(outputs)
        outputs = Add()([encoding_convolution_layers[index], deconvolution])

    number_of_channels = input_image_size[-1]

    outputs = Conv2D(filters=number_of_channels,
                     kernel_size=convolution_kernel_sizes[1],
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)

def create_denoising_auto_encoder_super_resolution_model_3d(input_image_size,
                                                            convolution_kernel_sizes=[(3, 3, 3), (5, 5, 5)],
                                                            number_of_encoding_layers=2,
                                                            number_of_filters=64
                                                           ):
    """
    2-D implementation of the denoising autoencoder image super resolution deep learning architecture.

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    convolution_kernel_sizes : list of 3-d tuples
        specifies the kernel size at each convolution layer.  Default values are
        the same as given in the original paper.  The length of kernel size list
        must be 1 greater than the tuple length of the number of filters.

    number_of_encoding_layers : integer
       The number of encoding layers.

    number_of_filters : integer
       The number of filters for each encoding layer.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_denoising_auto_encoder_super_resolution_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    inputs = Input(shape = input_image_size)

    outputs = inputs

    encoding_convolution_layers = []
    for i in range(number_of_encoding_layers):
        if i == 0:
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_sizes[0],
                             activation='relu',
                             padding='same')(outputs)
        else:
            layer = Conv3D(filters=number_of_filters,
                           kernel_size=convolution_kernel_sizes[0],
                           activation='relu',
                           padding='same')(outputs)
            encoding_convolution_layers.append(layer)

    outputs = encoding_convolution_layers[-1]

    for i in range(number_of_encoding_layers):
        index = len(encoding_convolution_layers) - i - 1
        deconvolution = Conv3DTranspose(filters=number_of_filters,
                                        kernel_size=convolution_kernel_sizes[0],
                                        padding='same',
                                        activation='relu')(outputs)
        outputs = Add()([encoding_convolution_layers[index], deconvolution])

    number_of_channels = input_image_size[-1]

    outputs = Conv3D(filters=number_of_channels,
                     kernel_size=convolution_kernel_sizes[1],
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)

