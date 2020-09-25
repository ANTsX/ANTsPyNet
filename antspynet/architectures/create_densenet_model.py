

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, BatchNormalization,
                                     Activation, Dense, Concatenate,
                                     Conv2D, Conv2DTranspose, GlobalAveragePooling2D,
                                     Conv3D, Conv3DTranspose, GlobalAveragePooling3D)

from tensorflow.keras import initializers
from tensorflow.keras import regularizers

def create_densenet_model_2d(input_image_size,
                             number_of_classification_labels=1000,
                             number_of_filters=16,
                             depth=7,
                             number_of_dense_blocks=1,
                             growth_rate=12,
                             dropout_rate=0.2,
                             weight_decay=1e-4,
                             mode='classification'
                            ):
    """
    2-D implementation of the Wide ResNet deep learning architecture.

    Creates a keras model of the DenseNet deep learning architecture for image
    recognition based on the paper

    G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected
    Convolutional Networks Networks

    available here:

            https://arxiv.org/abs/1608.06993

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    number_of_filters : integer
        Number of filters.

    depth : integer
        Number of layers---must be equal to 3 * N + 4 where N is an integer (default = 7).

    number_of_dense_blocks : integer
        Number of dense blocks number of dense blocks to add to the end (default = 1).

    growth_rate :  integer
        Number of filters to add for each dense block layer (default = 12).

    dropout_rate : scalar
        Per drop out layer rate (default = 0.2).

    weight_decay : scalar
        Weight decay (default = 1e-4).

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_densenet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    concatenation_axis = 0
    if K.image_data_format() == 'channels_last':
        concatenation_axis = -1

    def convolution_factory_2d(model, number_of_filters, kernel_size=(3, 3),
                               dropout_rate=0.0, weight_decay=1e-4):
        model = BatchNormalization(axis=concatenation_axis,
                                   gamma_regularizer=regularizers.l2(weight_decay),
                                   beta_regularizer=regularizers.l2(weight_decay))(model)
        model = Activation(activation='relu')(model)
        model = Conv2D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       padding='same',
                       use_bias=False,
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)
        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        return(model)

    def transition_2d(model, number_of_filters, dropout_rate=0.0, weight_decay=1e-4):

        model = convolution_factory_2d(model, number_of_filters, kernel_size=(1, 1),
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)
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

    number_of_layers = int((depth - 4) / 3)

    inputs = Input(shape = input_image_size)

    outputs = Conv2D(filters=number_of_filters,
                     kernel_size=(3, 3),
                     kernel_initializer='he_uniform',
                     padding='same',
                     use_bias=False,
                     kernel_regularizer=regularizers.l2(weight_decay))(inputs)

    # Add dense blocks

    nFilters = number_of_filters

    for i in range(number_of_dense_blocks - 1):
        outputs, nFilters = \
          create_dense_blocks_2d(outputs, number_of_filters=nFilters,
                                 depth=number_of_layers, growth_rate=growth_rate,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)
        outputs = transition_2d(outputs, number_of_filters=nFilters,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    outputs, nFilters = \
      create_dense_blocks_2d(outputs, number_of_filters=nFilters,
                             depth=number_of_layers, growth_rate=growth_rate,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)

    outputs = BatchNormalization(axis=concatenation_axis,
                                 gamma_regularizer=regularizers.l2(weight_decay),
                                 beta_regularizer=regularizers.l2(weight_decay))(outputs)


    outputs = Activation(activation='relu')(outputs)
    outputs = GlobalAveragePooling2D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation,
                    kernel_regularizer=regularizers.l2(weight_decay),
                    bias_regularizer=regularizers.l2(weight_decay))(outputs)

    densenet_model = Model(inputs=inputs, outputs=outputs)

    return(densenet_model)

def create_densenet_model_3d(input_image_size,
                             number_of_classification_labels=1000,
                             number_of_filters=16,
                             depth=7,
                             number_of_dense_blocks=1,
                             growth_rate=12,
                             dropout_rate=0.2,
                             weight_decay=1e-4,
                             mode='classification'
                            ):
    """
    2-D implementation of the Wide ResNet deep learning architecture.

    Creates a keras model of the DenseNet deep learning architecture for image
    recognition based on the paper

    G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected
    Convolutional Networks Networks

    available here:

            https://arxiv.org/abs/1608.06993

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    number_of_filters : integer
        Number of filters.

    depth : integer
        Number of layers---must be equal to 3 * N + 4 where N is an integer (default = 7).

    number_of_dense_blocks : integer
        Number of dense blocks number of dense blocks to add to the end (default = 1).

    growth_rate :  integer
        Number of filters to add for each dense block layer (default = 12).

    dropout_rate : scalar
        Per drop out layer rate (default = 0.2).

    weight_decay : scalar
        Weight decay (default = 1e-4).

    mode :  string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_densenet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    concatenation_axis = 0
    if K.image_data_format() == 'channels_last':
        concatenation_axis = -1

    def convolution_factory_3d(model, number_of_filters, kernel_size=(3, 3, 3),
                               dropout_rate=0.0, weight_decay=1e-4):
        model = BatchNormalization(axis=concatenation_axis,
                                   gamma_regularizer=regularizers.l2(weight_decay),
                                   beta_regularizer=regularizers.l2(weight_decay))(model)
        model = Activation(activation='relu')(model)
        model = Conv3D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       padding='same',
                       use_bias=False,
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)
        if dropout_rate > 0.0:
            model = Dropout(rate=dropout_rate)(model)

        return(model)

    def transition_3d(model, number_of_filters, dropout_rate=0.0, weight_decay=1e-4):

        model = convolution_factory_3d(model, number_of_filters, kernel_size=(1, 1, 1),
                                       dropout_rate=dropout_rate, weight_decay=weight_decay)
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

    number_of_layers = int((depth - 4) / 3)

    inputs = Input(shape = input_image_size)

    outputs = Conv3D(filters=number_of_filters,
                     kernel_size=(3, 3, 3),
                     kernel_initializer='he_uniform',
                     padding='same',
                     use_bias=False,
                     kernel_regularizer=regularizers.l2(weight_decay))(inputs)

    # Add dense blocks

    nFilters = number_of_filters

    for i in range(number_of_dense_blocks - 1):
        outputs, nFilters = \
          create_dense_blocks_3d(outputs, number_of_filters=nFilters,
                                 depth=number_of_layers, growth_rate=growth_rate,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)
        outputs = transition_3d(outputs, number_of_filters=nFilters,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)

    outputs, nFilters = \
      create_dense_blocks_3d(outputs, number_of_filters=nFilters,
                             depth=number_of_layers, growth_rate=growth_rate,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)

    outputs = BatchNormalization(axis=concatenation_axis,
                                 gamma_regularizer=regularizers.l2(weight_decay),
                                 beta_regularizer=regularizers.l2(weight_decay))(outputs)


    outputs = Activation(activation='relu')(outputs)
    outputs = GlobalAveragePooling3D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation,
                    kernel_regularizer=regularizers.l2(weight_decay),
                    bias_regularizer=regularizers.l2(weight_decay))(outputs)

    densenet_model = Model(inputs=inputs, outputs=outputs)

    return(densenet_model)



