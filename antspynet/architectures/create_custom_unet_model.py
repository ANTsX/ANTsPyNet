
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Add, Activation, BatchNormalization, Concatenate, ReLU, LeakyReLU,
                                     Conv3D, Conv3DTranspose, Input, Lambda, MaxPooling3D,
                                     SpatialDropout3D, UpSampling3D,
                                     Cropping2D, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D)

from tensorflow.keras.activations import softmax

from ..utilities import InstanceNormalization


def create_nobrainer_unet_model_3d(input_image_size):
    """
    Implementation of the "NoBrainer" U-net architecture

    Creates a keras model of the U-net deep learning architecture for image
    segmentation available at:

            https://github.com/neuronets/nobrainer/


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_nobrainer_unet_model_3d((None, None, None, 1))
    >>> model.summary()
    """

    number_of_outputs = 1
    number_of_filters_at_base_layer = 16
    convolution_kernel_size = (3, 3, 3)
    deconvolution_kernel_size = (2, 2, 2)

    inputs = Input(shape=input_image_size)

    # Encoding path

    outputs = Conv3D(filters=number_of_filters_at_base_layer,
                     kernel_size=convolution_kernel_size,
                     padding='same')(inputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip1 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip2 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip3 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*16,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    # Decoding path

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*16,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip3, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*8,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip2, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*4,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip1, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    convActivation = ''
    if number_of_outputs == 1:
        convActivation = 'sigmoid'
    else:
        convActivation = 'softmax'

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=1,
                     activation = convActivation)(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model

def create_hippmapp3r_unet_model_3d(input_image_size,
                                    do_first_network=True,
                                    data_format="channels_last"):
    """
    Implementation of the "HippMapp3r" U-net architecture

    Creates a keras model implementation of the u-net architecture
    described here:

        https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.24811

    with the implementation available here:

        https://github.com/mgoubran/HippMapp3r

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    do_first_network : boolean
        Boolean dictating if the model built should be the first (initial) network
        or second (refinement) network.

    data_format : string
        One of "channels_first" or "channels_last".  We do this for this specific
        architecture as the original weights were saved in "channels_first" format.

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> shape_initial_stage = (160, 160, 128)
    >>> model_initial_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_initial_stage, 1), True)
    >>> model_initial_stage.load_weights(antspynet.get_pretrained_network("hippMapp3rInitial"))
    >>> shape_refine_stage = (112, 112, 64)
    >>> model_refine_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), False)
    >>> model_refine_stage.load_weights(antspynet.get_pretrained_network("hippMapp3rRefine"))
    """

    channels_axis = None
    if data_format == "channels_last":
        channels_axis = 4
    elif data_format == "channels_first":
        channels_axis = 1
    else:
        raise ValueError("Unexpected string for data_format.")

    def convB_3d_layer(input, number_of_filters, kernel_size=3, strides=1):
        block = Conv3D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       data_format=data_format)(input)
        block = InstanceNormalization(axis=channels_axis)(block)
        block = LeakyReLU()(block)
        return(block)

    def residual_block_3d(input, number_of_filters):
        block = convB_3d_layer(input, number_of_filters)
        block = SpatialDropout3D(rate=0.3,
                                 data_format=data_format)(block)
        block = convB_3d_layer(block, number_of_filters)
        return(block)

    def upsample_block_3d(input, number_of_filters):
        block = UpSampling3D(data_format=data_format)(input)
        block = convB_3d_layer(block, number_of_filters)
        return(block)

    def feature_block_3d(input, number_of_filters):
        block = convB_3d_layer(input, number_of_filters)
        block = convB_3d_layer(block, number_of_filters, kernel_size=1)
        return(block)

    number_of_filters_at_base_layer = 16

    number_of_layers = 6
    if do_first_network == False:
        number_of_layers = 5

    inputs = Input(shape=input_image_size)

    # Encoding path

    add = None

    encoding_convolution_layers = []
    for i in range(number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2 ** i
        conv = None
        if i == 0:
            conv = convB_3d_layer(inputs, number_of_filters)
        else:
            conv = convB_3d_layer(add, number_of_filters, strides=2)
        residual_block = residual_block_3d(conv, number_of_filters)
        add = Add()([conv, residual_block])
        encoding_convolution_layers.append(add)

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers-1]

    # 256
    number_of_filters = (number_of_filters_at_base_layer *
      2 ** (number_of_layers - 2))
    outputs = upsample_block_3d(outputs, number_of_filters)

    if do_first_network == True:
        # 256, 128
        outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[4], outputs])
        outputs = feature_block_3d(outputs, number_of_filters)
        number_of_filters = int(number_of_filters / 2)
        outputs = upsample_block_3d(outputs, number_of_filters)

    # 128, 64
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[3], outputs])
    outputs = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(outputs, number_of_filters)

    # 64, 32
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[2], outputs])
    feature64 = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(feature64, number_of_filters)
    back64 = None
    if do_first_network == True:
        back64 = convB_3d_layer(feature64, 1, 1)
    else:
        back64 = Conv3D(filters=1,
                        kernel_size=1,
                        data_format=data_format)(feature64)
    back64 = UpSampling3D(data_format=data_format)(back64)

    # 32, 16
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[1], outputs])
    feature32 = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(feature32, number_of_filters)
    back32 = None
    if do_first_network == True:
        back32 = convB_3d_layer(feature32, 1, 1)
    else:
        back32 = Conv3D(filters=1,
                        kernel_size=1,
                        data_format=data_format)(feature32)
    back32 = Add()([back64, back32])
    back32 = UpSampling3D(data_format=data_format)(back32)

    # Final
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[0], outputs])
    outputs = convB_3d_layer(outputs, number_of_filters, 3)
    outputs = convB_3d_layer(outputs, number_of_filters, 1)
    if do_first_network == True:
        outputs = convB_3d_layer(outputs, 1, 1)
    else:
        outputs = Conv3D(filters=1,
                         kernel_size=1,
                         data_format=data_format)(outputs)
    outputs = Add()([back32, outputs])
    outputs = Activation('sigmoid')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return(unet_model)

def create_sysu_media_unet_model_2d(input_image_size,
                                    anatomy="wmh"):
    """
    Implementation of the sysu_media U-net architecture

    Creates a keras model implementation of the u-net architecture
    in the 2017 MICCAI WMH challenge by the sysu_medial team described
    here:

        https://pubmed.ncbi.nlm.nih.gov/30125711/

    with the original implementation available here:

        https://github.com/hongweilibran/wmh_ibbmTum

    Arguments
    ---------
    input_image_size : tuple of length 4
        This will be (200, 200, 2) for t1/flair input and (200, 200, 1)} for
        flair-only input.

    anatomy : string
        "wmh" or "claustrum"

    Returns
    -------
    Keras model
        A 2-D keras model defining the U-net network.

    Example
    -------
    >>> image_size = (200, 200)
    >>> model = antspynet.create_sysu_media_unet_model_2d((*image_size, 1))
    """

    def get_crop_shape(target_layer, reference_layer):

        delta = K.int_shape(target_layer)[1] - K.int_shape(reference_layer)[1]
        if delta % 2 != 0:
            cropShape0 = (int(delta/2), int(delta/2) + 1)
        else:
            cropShape0 = (int(delta/2), int(delta/2))

        delta = K.int_shape(target_layer)[2] - K.int_shape(reference_layer)[2]
        if delta % 2 != 0:
            cropShape1 = (int(delta/2), int(delta/2) + 1)
        else:
            cropShape1 = (int(delta/2), int(delta/2))

        return((cropShape0, cropShape1))

    inputs = Input(shape=input_image_size)

    if anatomy == "wmh":
        number_of_filters = (64, 96, 128, 256, 512)
    elif anatomy == "claustrum":
        number_of_filters = (32, 64, 96, 128, 256)

    # encoding layers

    encoding_layers = list()

    outputs = inputs
    for i in range(len(number_of_filters)):

        kernel1 = 3
        kernel2 = 3
        if i == 0 and anatomy == "wmh":
            kernel1 = 5
            kernel2 = 5
        elif i == 3:
            kernel1 = 3
            kernel2 = 4

        outputs = Conv2D(filters=number_of_filters[i],
                         kernel_size=kernel1,
                         padding='same')(outputs)
        outputs = Activation('relu')(outputs)
        outputs = Conv2D(filters=number_of_filters[i],
                         kernel_size=kernel2,
                         padding='same')(outputs)
        outputs = Activation('relu')(outputs)
        encoding_layers.append(outputs)
        if i < 4:
            outputs = MaxPooling2D(pool_size=(2, 2))(outputs)

    # decoding layers

    for i in range(len(encoding_layers)-2, -1, -1):
        upsample_layer = UpSampling2D(size=(2, 2))(outputs)
        crop_shape = get_crop_shape(encoding_layers[i], upsample_layer)
        cropped_layer = Cropping2D(cropping=crop_shape)(encoding_layers[i])
        outputs = Concatenate(axis=-1)([upsample_layer, cropped_layer])
        outputs = Conv2D(filters=number_of_filters[i],
                         kernel_size=3,
                         padding='same')(outputs)
        outputs = Activation('relu')(outputs)
        outputs = Conv2D(filters=number_of_filters[i],
                         kernel_size=3,
                         padding='same')(outputs)
        outputs = Activation('relu')(outputs)

    # final

    crop_shape = get_crop_shape(inputs, outputs)
    outputs = ZeroPadding2D(padding=crop_shape)(outputs)
    outputs = Conv2D(filters=1,
                     kernel_size=1,
                     activation='sigmoid',
                     padding='same')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return(unet_model)


def create_hypothalamus_unet_model_3d(input_image_size):

    """
    Implementation of the U-net architecture for hypothalamus segmentation
    described in

    https://pubmed.ncbi.nlm.nih.gov/32853816/

    and ported from the original implementation:

        https://github.com/BBillot/hypothalamus_seg

    The network has is characterized by the following parameters:
        * 3 resolution levels:  24 ---> 48 ---> 96 filters
        * convolution: kernel size:  (3, 3, 3), activation: 'elu',
        * pool size: (2, 2, 2)

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_hypothalamus_unet_model_3d((160, 160, 160, 1))
    """

    convolution_kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)
    number_of_outputs = 11

    number_of_layers = 3
    number_of_filters_at_base_layer = 24
    number_of_filters = list()
    for i in range(number_of_layers):
        number_of_filters.append(number_of_filters_at_base_layer * 2**i)

    inputs = Input(shape=(*input_image_size, 1))

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):

        if i == 0:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          activation='elu')(inputs)
        else:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          activation='elu')(pool)

        conv = Conv3D(filters=number_of_filters[i],
                      kernel_size=convolution_kernel_size,
                      padding='same',
                      activation='elu')(conv)

        encoding_convolution_layers.append(conv)

        conv = BatchNormalization(axis=-1)(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling3D(pool_size=pool_size)(conv)
        else:
            outputs = conv

    # Decoding path

    for i in range(1, number_of_layers):

        deconv = UpSampling3D(size=pool_size)(outputs)
        outputs = Concatenate(axis=4)([encoding_convolution_layers[number_of_layers-i-1], deconv])
        outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         activation='elu')(outputs)
        outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         activation='elu')(outputs)

        outputs = BatchNormalization(axis=-1)(outputs)

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=(1, 1, 1),
                     activation='softmax')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return(unet_model)
