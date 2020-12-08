
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Average, Conv2D, Conv3D)

def create_expanded_super_resolution_model_2d(input_image_size,
                                              convolution_kernel_sizes=[(9, 9), (1, 1), (3, 3), (5, 5), (5, 5)],
                                              number_of_filters=(64, 32, 32, 32)
                                             ):
    """
    2-D implementation of the expanded  image super resolution deep learning architecture.

    Creates a keras model of the image super resolution deep learning framework.
    based on the paper available here:

            https://arxiv.org/pdf/1501.00092

    This particular implementation is based on the following python
    implementation:

            https://github.com/titu1994/Image-Super-Resolution

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

    number_of_filters : tuple
       Contains the number of filters for each convolutional layer.  Default values
       are the same as given in the original paper.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_expanded_super_resolution_model_2d((128, 128, 1))
    >>> model.summary()
    """

    number_of_convolution_layers = len(convolution_kernel_sizes)

    if len(number_of_filters) != (number_of_convolution_layers - 1):
        raise ValueError("The length of the number of filters must be 1 less than the length of the convolution vector size")

    inputs = Input(shape = input_image_size)

    outputs = inputs

    averaging_convolution_layers = []
    for i in range(number_of_convolution_layers - 1):
        if i == 0:
            outputs = Conv2D(filters=number_of_filters[i],
                             kernel_size=convolution_kernel_sizes[i],
                             activation='relu',
                             padding='same')(outputs)
        else:
            layer = Conv2D(filters=number_of_filters[i],
                             kernel_size=convolution_kernel_sizes[i],
                             activation='relu',
                             padding='same')(outputs)
            averaging_convolution_layers.append(layer)

    outputs = Average()(averaging_convolution_layers)

    number_of_channels = input_image_size[-1]

    outputs = Conv2D(filters=number_of_channels,
                     kernel_size=convolution_kernel_sizes[-1],
                     activation='relu',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)


def create_expanded_super_resolution_model_3d(input_image_size,
                                              convolution_kernel_sizes=[(9, 9, 9), (1, 1, 1), (3, 3, 3), (5, 5, 5), (5, 5, 5)],
                                              number_of_filters=(64, 32, 32, 32)
                                             ):
    """
    3-D implementation of the expanded  image super resolution deep learning architecture.

    Creates a keras model of the image super resolution deep learning framework.
    based on the paper available here:

            https://arxiv.org/pdf/1501.00092

    This particular implementation is based on the following python
    implementation:

            https://github.com/titu1994/Image-Super-Resolution

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

    number_of_filters : tuple
       Contains the number of filters for each convolutional layer.  Default values
       are the same as given in the original paper.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_expanded_super_resolution_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    number_of_convolution_layers = len(convolution_kernel_sizes)

    if len(number_of_filters) != (number_of_convolution_layers - 1):
        raise ValueError("The length of the number of filters must be 1 less than the length of the convolution vector size")

    inputs = Input(shape = input_image_size)

    outputs = inputs

    averaging_convolution_layers = []
    for i in range(number_of_convolution_layers - 1):
        if i == 0:
            outputs = Conv3D(filters=number_of_filters[i],
                             kernel_size=convolution_kernel_sizes[i],
                             activation='relu',
                             padding='same')(outputs)
        else:
            layer = Conv3D(filters=number_of_filters[i],
                             kernel_size=convolution_kernel_sizes[i],
                             activation='relu',
                             padding='same')(outputs)
            averaging_convolution_layers.append(layer)

    outputs = Average()(averaging_convolution_layers)

    number_of_channels = input_image_size[-1]

    outputs = Conv3D(filters=number_of_channels,
                     kernel_size=convolution_kernel_sizes[-1],
                     activation='relu',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)
