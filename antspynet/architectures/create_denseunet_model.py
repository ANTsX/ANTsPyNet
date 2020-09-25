

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, BatchNormalization, Add,
                                     Concatenate, Dense, Activation,
                                     Conv2D, Conv2DTranspose, AveragePooling2D,
                                     MaxPooling2D, UpSampling2D, ZeroPadding2D,
                                     Conv3D, Conv3DTranspose, AveragePooling3D,
                                     MaxPooling3D, UpSampling3D, ZeroPadding3D)

from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from ..utilities import Scale

def create_denseunet_model_2d(input_image_size,
                              number_of_outputs=1,
                              number_of_layers_per_dense_block=(6, 12, 36, 24),
                              growth_rate=48,
                              initial_number_of_filters=96,
                              reduction_rate=0.0,
                              depth=7,
                              dropout_rate=0.0,
                              weight_decay=1e-4,
                              mode='classification'
                             ):
    """
    2-D implementation of the dense U-net deep learning architecture.

    Creates a keras model of the dense U-net deep learning architecture for
    image segmentation

    X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng. H-DenseUNet: Hybrid
    Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes

    available here:

            https://arxiv.org/pdf/1709.07330.pdf

    with the author's implementation available at:

            https://github.com/xmengli999/H-DenseUNet

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The
        shape (or dimension) of that tensor is the image dimensions followed by
        the number of channels (e.g., red, green, and blue).  The batch size
        (i.e., number of training images) is not specified a priori.

    number_of_outputs : integer
        Meaning depends on the mode.  For 'classification' this is the number of
        segmentation labels.  For 'regression' this is the number of outputs.

    number_of_layers_per_dense_blocks : tuple
        Number of dense blocks per layer.

    growth_rate : integer
        Number of filters to add for each dense block layer (default = 48).

    initial_number_of_filters : integer
        Number of filters at the beginning (default = 96).

    reduction_rate : scalar
        Reduction factor of transition blocks.

    depth :  integer
        Number of layers---must be equal to 3 * N + 4 where N is an integer
        (default = 7).

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers (default = 1e-4).

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_denseunet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    concatenation_axis=1
    if K.image_data_format() == 'channels_last':
        concatenation_axis=-1

    def convolution_factory_2d(model, number_of_filters,
                               kernel_size=(3, 3),
                               dropout_rate=0.0, weight_decay=1e-4):

        # Bottleneck layer

        model = BatchNormalization(axis=concatenation_axis)(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation('relu')(model)
        model = Conv2D(filters=(number_of_filters * 4),
                       kernel_size=(1, 1),
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        # Convolution layer

        model = BatchNormalization(axis=concatenation_axis,
                                   epsilon=1.1e-5)(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation(activation='relu')(model)
        model = ZeroPadding2D(padding=(1, 1))(model)
        model = Conv2D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        return(model)

    def transition_2d(model, number_of_filters, compression_rate=1.0,
                      dropout_rate=0.0, weight_decay=1e-4):

        model = BatchNormalization(axis=concatenation_axis,
                                   gamma_regularizer=regularizers.l2(weight_decay),
                                   beta_regularizer=regularizers.l2(weight_decay))(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation(activation='relu')(model)
        model = Conv2D(filters=int(number_of_filters * compression_rate),
                       kernel_size=(1, 1),
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        model = AveragePooling2D(pool_size=(2, 2),
                                 strides=(2, 2))(model)
        return(model)

    def create_dense_blocks_2d(model, number_of_filters, depth, growth_rate,
                               dropout_rate=0.0, weight_decay=1e-4):

        dense_block_layers = [model]
        for i in range(depth):
            model = convolution_factory_2d(model, number_of_filters=growth_rate,
                                           kernel_size=(3, 3), dropout_rate=dropout_rate,
                                           weight_decay=weight_decay)
            dense_block_layers.append(model)
            model = Concatenate(axis=concatenation_axis)(dense_block_layers)
            number_of_filters += growth_rate

        return(model, number_of_filters)


    if ((depth - 4) % 3) != 0:
        raise ValueError('Depth must be equal to 3*N+4 where N is an integer.')

    number_of_layers = int((depth - 4) % 3)
    number_of_dense_blocks = len(number_of_layers_per_dense_block)

    inputs = Input(shape = input_image_size)

    box_layers = []
    box_count = 1

    # Initial convolution

    outputs = ZeroPadding2D(padding=(3, 3))(inputs)
    outputs = Conv2D(filters=initial_number_of_filters,
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     use_bias=False)(outputs)
    outputs = BatchNormalization(epsilon=1.1e-5,
                                 axis=concatenation_axis)(outputs)
    outputs = Scale(axis=concatenation_axis)(outputs)
    outputs = Activation(activation='relu')(outputs)

    box_layers.append(outputs)
    box_count += 1

    outputs = ZeroPadding2D(padding=(1, 1))(outputs)
    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2))(outputs)

    # Add dense blocks

    nFilters = initial_number_of_filters

    for i in range(number_of_dense_blocks - 1):
        outputs, number_of_filters = \
           create_dense_blocks_2d(outputs, number_of_filters=nFilters,
                                  depth=number_of_layers_per_dense_block[i],
                                  growth_rate=growth_rate, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        box_layers.append(outputs)
        box_count += 1

        outputs = transition_2d(outputs, number_of_filters=number_of_filters,
                                compression_rate=(1.0 - reduction_rate),
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
        nFilters = int(number_of_filters * (1 - reduction_rate))


    outputs, nFilters = \
       create_dense_blocks_2d(outputs, number_of_filters=nFilters,
                              depth=number_of_layers_per_dense_block[number_of_dense_blocks - 1],
                              growth_rate=growth_rate, dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    outputs = BatchNormalization(epsilon=1.1e-5,
                                 axis=concatenation_axis)(outputs)
    outputs = Scale(axis=concatenation_axis)(outputs)
    outputs = Activation(activation='relu')(outputs)

    box_layers.append(outputs)
    box_count -= 1

    local_number_of_filters = (K.int_shape(box_layers[box_count]))[-1]
    local_layer = Conv2D(filters=local_number_of_filters,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='normal')(box_layers[box_count - 1])
    box_count -= 1

    for i in range(number_of_dense_blocks - 1):
        upsampling_layer = UpSampling2D(size=(2, 2))(outputs)
        outputs = Add()([local_layer, upsampling_layer])

        local_layer = box_layers[box_count - 1]
        box_count -= 1

        local_number_of_filters = (K.int_shape(box_layers[box_count]))[-1]
        outputs = Conv2D(filters=local_number_of_filters,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='normal')(outputs)

        if i == (number_of_dense_blocks - 2):
            outputs = Dropout(rate=0.3)(outputs)

        outputs = BatchNormalization(epsilon=1.1e-5,
                                     axis=concatenation_axis)(outputs)
        outputs = Activation(activation='relu')(outputs)

    convActivation = ''

    if mode == 'classification':
        convActivation = 'softmax'
    elif mode == 'regression':
        convActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Conv2D(filters=number_of_outputs,
                     kernel_size=(1, 1),
                     activation = convActivation,
                     kernel_initializer='normal')(outputs)

    denseunet_model = Model(inputs=inputs, outputs=outputs)

    return denseunet_model


def create_denseunet_model_3d(input_image_size,
                              number_of_outputs=1,
                              number_of_layers_per_dense_block=(6, 12, 36, 24),
                              growth_rate=48,
                              initial_number_of_filters=96,
                              reduction_rate=0.0,
                              depth=7,
                              dropout_rate=0.0,
                              weight_decay=1e-4,
                              mode='classification'
                             ):
    """
    2-D implementation of the dense U-net deep learning architecture.

    Creates a keras model of the dense U-net deep learning architecture for
    image segmentation

    X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng. H-DenseUNet: Hybrid
    Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes

    available here:

            https://arxiv.org/pdf/1709.07330.pdf

    with the author's implementation available at:

            https://github.com/xmengli999/H-DenseUNet

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The
        shape (or dimension) of that tensor is the image dimensions followed by
        the number of channels (e.g., red, green, and blue).  The batch size
        (i.e., number of training images) is not specified a priori.

    number_of_outputs : integer
        Meaning depends on the mode.  For 'classification' this is the number of
        segmentation labels.  For 'regression' this is the number of outputs.

    number_of_layers_per_dense_blocks : tuple
        Number of dense blocks per layer.

    growth_rate : integer
        Number of filters to add for each dense block layer (default = 48).

    initial_number_of_filters : integer
        Number of filters at the beginning (default = 96).

    reduction_rate : scalar
        Reduction factor of transition blocks.

    depth :  integer
        Number of layers---must be equal to 3 * N + 4 where N is an integer
        (default = 7).

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers (default = 1e-4).

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_denseunet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    concatenation_axis=1
    if K.image_data_format() == 'channels_last':
        concatenation_axis=-1

    def convolution_factory_3d(model, number_of_filters,
                               kernel_size=(3, 3, 3),
                               dropout_rate=0.0, weight_decay=1e-4):

        # Bottleneck layer

        model = BatchNormalization(axis=concatenation_axis)(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation('relu')(model)
        model = Conv3D(filters=(number_of_filters * 4),
                       kernel_size=(1, 1, 1),
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        # Convolution layer

        model = BatchNormalization(axis=concatenation_axis,
                                   epsilon=1.1e-5)(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation(activation='relu')(model)
        model = ZeroPadding3D(padding=(1, 1, 1))(model)
        model = Conv3D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        return(model)

    def transition_3d(model, number_of_filters, compression_rate=1.0,
                      dropout_rate=0.0, weight_decay=1e-4):

        model = BatchNormalization(axis=concatenation_axis,
                                   gamma_regularizer=regularizers.l2(weight_decay),
                                   beta_regularizer=regularizers.l2(weight_decay))(model)
        model = Scale(axis=concatenation_axis)(model)
        model = Activation(activation='relu')(model)
        model = Conv3D(filters=int(number_of_filters * compression_rate),
                       kernel_size=(1, 1, 1),
                       use_bias=False)(model)

        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        model = AveragePooling3D(pool_size=(2, 2, 2),
                                 strides=(2, 2, 2))(model)
        return(model)

    def create_dense_blocks_3d(model, number_of_filters, depth, growth_rate,
                               dropout_rate=0.0, weight_decay=1e-4):

        dense_block_layers = [model]
        for i in range(depth):
            model = convolution_factory_3d(model, number_of_filters=growth_rate,
                                           kernel_size=(3, 3, 3), dropout_rate=dropout_rate,
                                           weight_decay=weight_decay)
            dense_block_layers.append(model)
            model = Concatenate(axis=concatenation_axis)(dense_block_layers)
            number_of_filters += growth_rate

        return(model, number_of_filters)


    if ((depth - 4) % 3) != 0:
        raise ValueError('Depth must be equal to 3*N+4 where N is an integer.')

    number_of_layers = int((depth - 4) % 3)
    number_of_dense_blocks = len(number_of_layers_per_dense_block)

    inputs = Input(shape = input_image_size)

    box_layers = []
    box_count = 1

    # Initial convolution

    outputs = ZeroPadding3D(padding=(3, 3))(inputs)
    outputs = Conv3D(filters=initial_number_of_filters,
                     kernel_size=(7, 7, 7),
                     strides=(2, 2, 2),
                     use_bias=False)(outputs)
    outputs = BatchNormalization(epsilon=1.1e-5,
                                 axis=concatenation_axis)(outputs)
    outputs = Scale(axis=concatenation_axis)(outputs)
    outputs = Activation(activation='relu')(outputs)

    box_layers.append(outputs)
    box_count += 1

    outputs = ZeroPadding3D(padding=(1, 1, 1))(outputs)
    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2))(outputs)

    # Add dense blocks

    nFilters = initial_number_of_filters

    for i in range(number_of_dense_blocks - 1):
        outputs, number_of_filters = \
           create_dense_blocks_3d(outputs, number_of_filters=nFilters,
                                  depth=number_of_layers_per_dense_block[i],
                                  growth_rate=growth_rate, dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        box_layers.append(outputs)
        box_count += 1

        outputs = transition_3d(outputs, number_of_filters=number_of_filters,
                                compression_rate=(1.0 - reduction_rate),
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
        nFilters = int(number_of_filters * (1 - reduction_rate))


    outputs, nFilters = \
       create_dense_blocks_3d(outputs, number_of_filters=nFilters,
                              depth=number_of_layers_per_dense_block[number_of_dense_blocks - 1],
                              growth_rate=growth_rate, dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    outputs = BatchNormalization(epsilon=1.1e-5,
                                 axis=concatenation_axis)(outputs)
    outputs = Scale(axis=concatenation_axis)(outputs)
    outputs = Activation(activation='relu')(outputs)

    box_layers.append(outputs)
    box_count -= 1

    local_number_of_filters = (K.int_shape(box_layers[box_count]))[-1]
    local_layer = Conv3D(filters=local_number_of_filters,
                         kernel_size=(1, 1, 1),
                         padding='same',
                         kernel_initializer='normal')(box_layers[box_count - 1])
    box_count -= 1

    for i in range(number_of_dense_blocks - 1):
        upsampling_layer = UpSampling3D(size=(2, 2, 2))(outputs)
        outputs = Add()([local_layer, upsampling_layer])

        local_layer = box_layers[box_count - 1]
        box_count -= 1

        local_number_of_filters = (K.int_shape(box_layers[box_count]))[-1]
        outputs = Conv3D(filters=local_number_of_filters,
                         kernel_size=(3, 3, 3),
                         padding='same',
                         kernel_initializer='normal')(outputs)

        if i == (number_of_dense_blocks - 2):
            outputs = Dropout(rate=0.3)(outputs)

        outputs = BatchNormalization(epsilon=1.1e-5,
                                     axis=concatenation_axis)(outputs)
        outputs = Activation(activation='relu')(outputs)

    convActivation = ''

    if mode == 'classification':
        convActivation = 'softmax'
    elif mode == 'regression':
        convActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=(1, 1, 1),
                     activation = convActivation,
                     kernel_initializer='normal')(outputs)

    denseunet_model = Model(inputs=inputs, outputs=outputs)

    return denseunet_model


