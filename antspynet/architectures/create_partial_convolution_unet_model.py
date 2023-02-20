import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (BatchNormalization, Concatenate, ReLU,
                                     Conv3D, Conv3DTranspose, Input, Lambda, MaxPooling3D,
                                     ReLU, UpSampling3D,
                                     Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D)

from ..utilities import ResampleTensorLayer2D, ResampleTensorLayer3D

def create_partial_convolution_unet_model_2d(input_image_size,
                                             number_of_priors=0,
                                             number_of_filters=(64, 128, 256, 512, 512, 512, 512, 512),
                                             kernel_size=(7, 5, 5, 3, 3, 3, 3, 3),
                                             use_partial_conv=True):

    """
    2-D implementation of the U-net architecture for inpainting using partial
    convolution.

        https://arxiv.org/abs/1804.07723

    Arguments
    ---------
    input_image_size : tuple of length 3
        Tuple of ints of length 3 specifying 2-D image size and channel size.

    number_of_priors : int
        Specify tissue priors for use during the decoding branch.

    number_of_filters: tuple
        Specifies the filter schedule.  Defaults to the number of filters used in
        the paper.

    kernel_size: single scalar or tuple of same length as the number of filters.
        Specifies the kernel size schedule for the encoding path.  Defaults to the
        kernel sizes used in the paper.

    use_partial_conv:  boolean
        Testing.  Switch between vanilla convolution layers and partial convolution layers.

    Returns
    -------
    Keras model
        A 2-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_partial_convolution_unet_model_2d((256, 256, 1)))
    """

    from ..utilities import PartialConv2D

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(number_of_filters)
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0]] * len(number_of_filters)
    elif len(kernel_size) != len(number_of_filters):
        raise ValueError("kernel_size must be a scalar or of equal length as the number_of_filters.")

    input_image = Input(input_image_size)
    input_mask = Input(input_image_size)

    if number_of_priors > 0:
        input_priors = Input((input_image_size[0], input_image_size[1], number_of_priors))
        inputs = [input_image, input_mask, input_priors]
    else:
        inputs = [input_image, input_mask]

    # Encoding path

    number_of_layers = len(number_of_filters)

    encoding_convolution_layers = []
    pool = None
    mask = None
    for i in range(number_of_layers):

        if i == 0:
            if use_partial_conv:
                conv, mask = PartialConv2D(filters=number_of_filters[i],
                                           kernel_size=kernel_size[i],
                                           padding="same")([inputs[0], inputs[1]])
            else:
                conv = Conv2D(filters=number_of_filters[i],
                              kernel_size=kernel_size[i],
                              padding='same')(inputs[0])
        else:
            if use_partial_conv:
                mask = ResampleTensorLayer2D(shape=(pool.shape[1], pool.shape[2]),
                                             interpolation_type='nearest_neighbor')(mask)
                conv, mask = PartialConv2D(filters=number_of_filters[i],
                                           kernel_size=kernel_size[i],
                                           padding="same")([pool, mask])
            else:
                conv = Conv2D(filters=number_of_filters[i],
                              kernel_size=kernel_size[i],
                              padding='same')(pool)
        conv = ReLU()(conv)

        if use_partial_conv:
            conv, mask = PartialConv2D(filters=number_of_filters[i],
                                       kernel_size=kernel_size[i],
                                       padding="same")([conv, mask])
        else:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=kernel_size[i],
                          padding='same')(conv)
        conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling2D(pool_size=(2,2),
                                strides=(2,2))(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        deconv = Conv2DTranspose(filters=number_of_filters[number_of_layers-i-1],
                                 kernel_size=2,
                                 padding='same')(outputs)
        deconv = UpSampling2D(size=(2,2))(deconv)

        if use_partial_conv:
            mask = UpSampling2D(size=(2,2),
                                interpolation="nearest")(mask)

        outputs = Concatenate(axis=3)([deconv, encoding_convolution_layers[number_of_layers-i-1]])
        if use_partial_conv:
            mask = Lambda(lambda x: tf.repeat(tf.gather(x[0], [0], axis=-1), tf.shape(x[1])[-1], axis=-1))([mask, outputs])

        if number_of_priors > 0:
            resampled_priors = ResampleTensorLayer2D(shape=(outputs.shape[1], outputs.shape[2]),
                                                     interpolation_type='linear')(input_priors)
            outputs = Concatenate(axis=3)([outputs, resampled_priors])
            if use_partial_conv:
                resampled_priors_mask = Lambda(lambda x: tf.ones_like(x))(resampled_priors)
                mask = Concatenate(axis=3)([mask, resampled_priors_mask])

        if use_partial_conv:
            outputs, mask = PartialConv2D(filters=number_of_filters[number_of_layers-i-1],
                                       kernel_size=3,
                                       padding="same")([outputs, mask])
        else:
            outputs = Conv2D(filters=number_of_filters[number_of_layers-i-1],
                             kernel_size=3,
                             padding='same')(outputs)
        outputs = ReLU()(outputs)

        if use_partial_conv:
            outputs, mask = PartialConv2D(filters=number_of_filters[number_of_layers-i-1],
                                       kernel_size=3,
                                       padding="same")([outputs, mask])
        else:
            outputs = Conv2D(filters=number_of_filters[number_of_layers-i-1],
                             kernel_size=3,
                             padding='same')(outputs)
        outputs = ReLU()(outputs)

    outputs = Conv2D(filters=1,
                     kernel_size=(1, 1),
                     activation = 'linear')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


def create_partial_convolution_unet_model_3d(input_image_size,
                                             number_of_priors=0,
                                             number_of_filters=(64, 128, 256, 512, 512, 512, 512, 512),
                                             kernel_size=(7, 5, 5, 3, 3, 3, 3, 3),
                                             use_partial_conv=True):

    """
    3-D implementation of the U-net architecture for inpainting using partial
    convolution.

        https://arxiv.org/abs/1804.07723

    Arguments
    ---------
    input_image_size : tuple of length 3
        Tuple of ints of length 3 specifying 2-D image size and channel size.

    number_of_priors : int
        Specify tissue priors for use during the decoding branch.

    number_of_filters: tuple
        Specifies the filter schedule.  Defaults to the number of filters used in
        the paper.

    kernel_size: single scalar or tuple of same length as the number of filters.
        Specifies the kernel size schedule for the encoding path.  Defaults to the
        kernel sizes used in the paper.

    use_partial_conv:  boolean
        Testing.  Switch between vanilla convolution layers and partial convolution layers.

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_partial_convolution_unet_model_3d((256, 256, 256, 1)))
    """

    from ..utilities import PartialConv3D

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size] * len(number_of_filters)
    elif len(kernel_size) == 1:
        kernel_size = [kernel_size[0]] * len(number_of_filters)
    elif len(kernel_size) != len(number_of_filters):
        raise ValueError("kernel_size must be a scalar or of equal length as the number_of_filters.")

    input_image = Input(input_image_size)
    input_mask = Input(input_image_size)

    if number_of_priors > 0:
        input_priors = Input((input_image_size[0], input_image_size[1], input_image_size[2], number_of_priors))
        inputs = [input_image, input_mask, input_priors]
    else:
        inputs = [input_image, input_mask]

    # Encoding path

    number_of_layers = len(number_of_filters)

    encoding_convolution_layers = []
    pool = None
    mask = None
    for i in range(number_of_layers):

        if i == 0:
            if use_partial_conv:
                conv, mask = PartialConv3D(filters=number_of_filters[i],
                                           kernel_size=kernel_size[i],
                                           padding="same")([inputs[0], inputs[1]])
            else:
                conv = Conv3D(filters=number_of_filters[i],
                              kernel_size=kernel_size[i],
                              padding='same')(inputs[0])
        else:
            if use_partial_conv:
                mask = ResampleTensorLayer3D(shape=(pool.shape[1], pool.shape[2], pool.shape[3]),
                                             interpolation_type='nearest_neighbor')(mask)
                conv, mask = PartialConv3D(filters=number_of_filters[i],
                                           kernel_size=kernel_size[i],
                                           padding="same")([pool, mask])
            else:
                conv = Conv3D(filters=number_of_filters[i],
                              kernel_size=kernel_size[i],
                              padding='same')(pool)
        conv = ReLU()(conv)

        if use_partial_conv:
            conv, mask = PartialConv3D(filters=number_of_filters[i],
                                       kernel_size=kernel_size[i],
                                       padding="same")([conv, mask])
        else:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=kernel_size[i],
                          padding='same')(conv)
        conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling3D(pool_size=(2,2,2),
                                strides=(2,2,2))(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        deconv = Conv3DTranspose(filters=number_of_filters[number_of_layers-i-1],
                                 kernel_size=2,
                                 padding='same')(outputs)
        deconv = UpSampling3D(size=(2,2,2))(deconv)

        if use_partial_conv:
            mask = UpSampling3D(size=(2,2,2))(mask)

        outputs = Concatenate(axis=4)([deconv, encoding_convolution_layers[number_of_layers-i-1]])
        if use_partial_conv:
            mask = Lambda(lambda x: tf.repeat(tf.gather(x[0], [0], axis=-1), tf.shape(x[1])[-1], axis=-1))([mask, outputs])

        if number_of_priors > 0:
            resampled_priors = ResampleTensorLayer3D(shape=(outputs.shape[1], outputs.shape[2], outputs.shape[3]),
                                                     interpolation_type='linear')(input_priors)
            outputs = Concatenate(axis=4)([outputs, resampled_priors])
            if use_partial_conv:
                resampled_priors_mask = Lambda(lambda x: tf.ones_like(x))(resampled_priors)
                mask = Concatenate(axis=4)([mask, resampled_priors_mask])

        if use_partial_conv:
            outputs, mask = PartialConv3D(filters=number_of_filters[number_of_layers-i-1],
                                       kernel_size=3,
                                       padding="same")([outputs, mask])
        else:
            outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                             kernel_size=3,
                             padding='same')(outputs)
        outputs = ReLU()(outputs)

        if use_partial_conv:
            outputs, mask = PartialConv3D(filters=number_of_filters[number_of_layers-i-1],
                                       kernel_size=3,
                                       padding="same")([outputs, mask])
        else:
            outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                             kernel_size=3,
                             padding='same')(outputs)
        outputs = ReLU()(outputs)

    outputs = Conv3D(filters=1,
                     kernel_size=(1, 1, 1),
                     activation = 'linear')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model
