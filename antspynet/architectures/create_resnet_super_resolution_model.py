
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Activation, Add, BatchNormalization,
                          Conv2D, Conv2DTranspose,
                          MaxPooling2D, UpSampling2D,
                          Conv3D, Conv3DTranspose,
                          MaxPooling3D, UpSampling3D)

def create_resnet_super_resolution_model_2d(input_image_size,
                                            convolution_kernel_size=(3, 3),
                                            number_of_filters=64,
                                            number_of_residual_blocks=5,
                                            number_of_resnet_blocks=1
                                           ):
    """
    2-D implementation of the ResNet image super resolution architecture.

    Creates a keras model of the expanded image super resolution deep learning
    framework based on the following python implementation:

            https://github.com/titu1994/Image-Super-Resolution

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    convolution_kernel_size : 2-d tuple
        Specifies the kernel size

    number_of_filters : integer
        The number of filters for each encoding layer.

    number_of_residual_blocks : integer
        Number of residual blocks.

    number_of_resnet_blocks : integer
        Number of resnet blocks.  Each block will double the upsampling amount.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_resnet_super_resolution_model_2d((128, 128, 1))
    >>> model.summary()
    """

    def residual_block_2d(model, number_of_filters, convolution_kernel_size):

        block = Conv2D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='linear',
                       padding='same')(model)
        block = BatchNormalization()(block)
        block = Activation(activation='relu')(block)
        block = Conv2D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='linear',
                       padding='same')(block)
        block = BatchNormalization()(block)
        block = Add()([model, block])

        return(block)

    def upscale_block_2d(model, number_of_filters, convolution_kernel_size):

        block = UpSampling2D()(model)
        block = Conv2D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='relu',
                       padding='same')(block)
        return(block)

    def resnet_block_2d(inputs, number_of_filters, convolution_kernel_size,
                        number_of_residual_blocks):

        outputs = Conv2D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same')(inputs)

        residual_blocks = residual_block_2d(outputs, number_of_filters,
                                            convolution_kernel_size)

        for i in range(number_of_residual_blocks):
            residual_blocks = residual_block_2d(residual_blocks, number_of_filters,
                                                convolution_kernel_size)

        outputs = Add()([residual_blocks, outputs])
        outputs = upscale_block_2d(outputs, number_of_filters, convolution_kernel_size)
        return(outputs)

    inputs = Input(shape = input_image_size)

    outputs = resnet_block_2d(inputs, number_of_filters, convolution_kernel_size,
                              number_of_residual_blocks)
    if number_of_resnet_blocks > 1:
        for i in range(1, number_of_resnet_blocks):
             outputs = resnet_block_2d(outputs, number_of_filters,
                                       convolution_kernel_size, number_of_residual_blocks)

    number_of_channels = input_image_size[-1]

    outputs = Conv2D(filters=number_of_channels,
                     kernel_size=convolution_kernel_size,
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)


def create_resnet_super_resolution_model_3d(input_image_size,
                                            convolution_kernel_size=(3, 3, 3),
                                            number_of_filters=64,
                                            number_of_residual_blocks=5,
                                            number_of_resnet_blocks=1
                                           ):
    """
    3-D implementation of the ResNet image super resolution architecture.

    Creates a keras model of the expanded image super resolution deep learning
    framework based on the following python implementation:

            https://github.com/titu1994/Image-Super-Resolution

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    convolution_kernel_size : 3-d tuple
        Specifies the kernel size

    number_of_filters : integer
        The number of filters for each encoding layer.

    number_of_residual_blocks : integer
        Number of residual blocks.

    number_of_resnet_blocks : integer
        Number of resnet blocks.  Each block will double the upsampling amount.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_resnet_super_resolution_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    def residual_block_3d(model, number_of_filters, convolution_kernel_size):

        block = Conv3D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='linear',
                       padding='same')(model)
        block = BatchNormalization()(block)
        block = Activation(activation='relu')(block)
        block = Conv3D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='linear',
                       padding='same')(block)
        block = BatchNormalization()(block)
        block = Add()([model, block])

        return(block)

    def upscale_block_3d(model, number_of_filters, convolution_kernel_size):

        block = UpSampling3D()(model)
        block = Conv3D(filters=number_of_filters,
                       kernel_size=convolution_kernel_size,
                       activation='relu',
                       padding='same')(block)
        return(block)

    def resnet_block_3d(inputs, number_of_filters, convolution_kernel_size,
                        number_of_residual_blocks):

        outputs = Conv3D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same')(inputs)

        residual_blocks = residual_block_3d(outputs, number_of_filters,
                                            convolution_kernel_size)

        for i in range(number_of_residual_blocks):
            residual_blocks = residual_block_3d(residual_blocks, number_of_filters,
                                                convolution_kernel_size)

        outputs = Add()([residual_blocks, outputs])
        outputs = upscale_block_3d(outputs, number_of_filters, convolution_kernel_size)
        return(outputs)

    inputs = Input(shape = input_image_size)

    outputs = resnet_block_3d(inputs, number_of_filters, convolution_kernel_size,
                              number_of_residual_blocks)
    if number_of_resnet_blocks > 1:
        for i in range(1, number_of_resnet_blocks):
             outputs = resnet_block_3d(outputs, number_of_filters,
                                       convolution_kernel_size, number_of_residual_blocks)

    number_of_channels = input_image_size[-1]

    outputs = Conv3D(filters=number_of_channels,
                     kernel_size=convolution_kernel_size,
                     activation='linear',
                     padding='same')(outputs)

    sr_model = Model(inputs=inputs, outputs=outputs)

    return(sr_model)


