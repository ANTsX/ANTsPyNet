
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, BatchNormalization, Add,
                                     Activation, Dense, Flatten,
                                     Conv2D, Conv2DTranspose, AveragePooling2D,
                                     Conv3D, Conv3DTranspose, AveragePooling3D)

def create_wide_resnet_model_2d(input_image_size,
                                number_of_classification_labels=1000,
                                depth=2,
                                width=1,
                                residual_block_schedule=(16, 32, 64),
                                pool_size=(8, 8),
                                dropout_rate=0.0,
                                weight_decay=0.0005,
                                mode='classification'
                               ):
    """
    2-D implementation of the Wide ResNet deep learning architecture.

    Creates a keras model of the Wide ResNet deep learning architecture for image
    classification/regression.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/titu1994/Wide-Residual-Networks

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    depth : integer
        Determines the depth of the network.  Related to the actual number of
        layers by number_of_layers = depth * 6 + 4.    Default = 2 (such that
        number_of_layers = 16).

    width : integer
        Determines the width of the network.  Default = 1.

    residual_block_schedule : tuple
        Determines the number of filters per convolutional block. Default =
        (16, 32, 64).

    pool_size : tuple
        Pool size for final average pooling layer.  Default = (8, 8).

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for regularization of the kernel weights of the
        convolution layers.  Default = 0.0005.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_wide_resnet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    channel_axis = 0
    if K.image_data_format() == 'channels_last':
        channel_axis = -1

    def initial_convolution_layer(model, number_of_filters):
        model = Conv2D(filters=number_of_filters,
                       kernel_size=(3, 3),
                       padding='same',
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)
        model = BatchNormalization(axis=channel_axis,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer="uniform")(model)
        model = Activation(activation='relu')(model)

        return(model)

    def custom_convolution_layer(initial_model, base, width, strides=(1, 1),
                                 dropout_rate=0.0, expand=True):
        number_of_filters=int(base * width)

        if expand:
            model = Conv2D(filters=number_of_filters,
                           kernel_size=(3, 3),
                           padding='same',
                           strides=strides,
                           use_bias=False,
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(weight_decay))(initial_model)
        else:
            model = initial_model

        model = BatchNormalization(axis=channel_axis,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer="uniform")(model)
        model = Activation(activation='relu')(model)

        model = Conv2D(filters=number_of_filters,
                       kernel_size=(3, 3),
                       padding='same',
                       use_bias=False,
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)

        if expand:
            skip_layer = Conv2D(filters=number_of_filters,
                                kernel_size=(1, 1),
                                padding='same',
                                strides=strides,
                                use_bias=False,
                                kernel_initializer=initializers.he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay))(initial_model)
            model = Add()([model, skip_layer])
        else:
            if dropout_rate > 0.0:
                model = Dropout(rate=dropout_rate)(model)

            model = BatchNormalization(axis=channel_axis,
                                       momentum=0.1,
                                       epsilon=1e-5,
                                       gamma_initializer="uniform")(model)
            model = Activation(activation='relu')(model)

            model = Conv2D(filters=number_of_filters,
                           kernel_size=(3, 3),
                           padding='same',
                           use_bias=False,
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(weight_decay))(model)

            model = Add()([initial_model, model])

        return(model)

    inputs = Input(shape=input_image_size)

    outputs = initial_convolution_layer(inputs, residual_block_schedule[0])
    number_of_convolutions = 4

    for i in range(len(residual_block_schedule)):
        base_number_of_filters = residual_block_schedule[i]

        outputs = custom_convolution_layer(outputs, base = base_number_of_filters,
                                           width=width, strides=(1, 1), dropout_rate=0.0,
                                           expand=True)
        number_of_convolutions += 2

        for j in range(depth):
            outputs = custom_convolution_layer(outputs, base = base_number_of_filters,
                                               width=width, dropout_rate=dropout_rate,
                                               expand=False)
            number_of_convolutions += 2

        outputs = BatchNormalization(axis=channel_axis,
                                     momentum=0.1,
                                     epsilon=1e-5,
                                     gamma_initializer="uniform")(outputs)
        outputs = Activation(activation='relu')(outputs)

    outputs = AveragePooling2D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation,
                    kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    wide_resnet_model = Model(inputs=inputs, outputs=outputs)

    return(wide_resnet_model)


def create_wide_resnet_model_3d(input_image_size,
                                number_of_classification_labels=1000,
                                depth=2,
                                width=1,
                                residual_block_schedule=(16, 32, 64),
                                pool_size=(8, 8, 8),
                                dropout_rate=0.0,
                                weight_decay=0.0005,
                                mode='classification'
                               ):
    """
    3-D implementation of the Wide ResNet deep learning architecture.

    Creates a keras model of the Wide ResNet deep learning architecture for image
    classification/regression.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/titu1994/Wide-Residual-Networks

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    depth : integer
        Determines the depth of the network.  Related to the actual number of
        layers by number_of_layers = depth * 6 + 4.    Default = 2 (such that
        number_of_layers = 16).

    width : integer
        Determines the width of the network.  Default = 1.

    residual_block_schedule : tuple
        Determines the number of filters per convolutional block. Default =
        (16, 32, 64).

    pool_size : tuple
        Pool size for final average pooling layer.  Default = (8, 8, 8).

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for regularization of the kernel weights of the
        convolution layers.  Default = 0.0005.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_wide_resnet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    channel_axis = 0
    if K.image_data_format() == 'channels_last':
        channel_axis = -1

    def initial_convolution_layer(model, number_of_filters):
        model = Conv3D(filters=number_of_filters,
                       kernel_size=(3, 3, 3),
                       padding='same',
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)
        model = BatchNormalization(axis=channel_axis,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer="uniform")(model)
        model = Activation(activation='relu')(model)

        return(model)

    def custom_convolution_layer(initial_model, base, width, strides=(1, 1, 1),
                                 dropout_rate=0.0, expand=True):
        number_of_filters=int(base * width)

        if expand:
            model = Conv3D(filters=number_of_filters,
                           kernel_size=(3, 3, 3),
                           padding='same',
                           strides=strides,
                           use_bias=False,
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(weight_decay))(initial_model)
        else:
            model = initial_model

        model = BatchNormalization(axis=channel_axis,
                                   momentum=0.1,
                                   epsilon=1e-5,
                                   gamma_initializer="uniform")(model)
        model = Activation(activation='relu')(model)

        model = Conv3D(filters=number_of_filters,
                       kernel_size=(3, 3, 3),
                       padding='same',
                       use_bias=False,
                       kernel_initializer=initializers.he_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(model)

        if expand:
            skip_layer = Conv3D(filters=number_of_filters,
                                kernel_size=(1, 1, 1),
                                padding='same',
                                strides=strides,
                                use_bias=False,
                                kernel_initializer=initializers.he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay))(initial_model)
            model = Add()([model, skip_layer])
        else:
            if dropout_rate > 0.0:
                model = Dropout(rate=dropout_rate)(model)

            model = BatchNormalization(axis=channel_axis,
                                       momentum=0.1,
                                       epsilon=1e-5,
                                       gamma_initializer="uniform")(model)
            model = Activation(activation='relu')(model)

            model = Conv3D(filters=number_of_filters,
                           kernel_size=(3, 3, 3),
                           padding='same',
                           use_bias=False,
                           kernel_initializer=initializers.he_normal(),
                           kernel_regularizer=regularizers.l2(weight_decay))(model)

            model = Add()([initial_model, model])

        return(model)

    inputs = Input(shape=input_image_size)

    outputs = initial_convolution_layer(inputs, residual_block_schedule[0])
    number_of_convolutions = 4

    for i in range(len(residual_block_schedule)):
        base_number_of_filters = residual_block_schedule[i]

        outputs = custom_convolution_layer(outputs, base = base_number_of_filters,
                                           width=width, strides=(1, 1, 1), dropout_rate=0.0,
                                           expand=True)
        number_of_convolutions += 2

        for j in range(depth):
            outputs = custom_convolution_layer(outputs, base = base_number_of_filters,
                                               width=width, dropout_rate=dropout_rate,
                                               expand=False)
            number_of_convolutions += 2

        outputs = BatchNormalization(axis=channel_axis,
                                     momentum=0.1,
                                     epsilon=1e-5,
                                     gamma_initializer="uniform")(outputs)
        outputs = Activation(activation='relu')(outputs)

    outputs = AveragePooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation,
                    kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    wide_resnet_model = Model(inputs=inputs, outputs=outputs)

    return(wide_resnet_model)

