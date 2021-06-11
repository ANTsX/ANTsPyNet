
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, BatchNormalization, Add,
                                     LeakyReLU, Concatenate, Lambda, Dense,
                                     Reshape, Permute, Multiply,
                                     Conv2D, Conv2DTranspose,
                                     MaxPooling2D, GlobalAveragePooling2D,
                                     UpSampling2D,
                                     Conv3D, Conv3DTranspose,
                                     MaxPooling3D, GlobalAveragePooling3D,
                                     UpSampling3D)

def create_resnet_model_2d(input_image_size,
                           input_scalars_size=0,
                           number_of_classification_labels=1000,
                           layers=(1, 2, 3, 4),
                           residual_block_schedule=(3, 4, 6, 3),
                           lowest_resolution=64,
                           cardinality=1,
                           squeeze_and_excite=False,
                           mode='classification'
                          ):
    """
    2-D implementation of the ResNet deep learning architecture.

    Creates a keras model of the ResNet deep learning architecture for image
    classification.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    input_scalars_size : integer
        Optional integer specifying the size of the input vector for scalars that
        get concatenated to the fully connected layer at the end of the network.

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    residual_block_schedule : tuple
        A tuple defining the how many residual blocks repeats for each layer.

    lowest_resolution : integer
        Number of filters at the initial layer.

    cardinality : integer
        perform ResNet (cardinality = 1) or ResNeX (cardinality does not 1 but,
        instead, powers of 2---try '32').

    squeeze_and_excite : boolean
        add the squeeze-and-excite block variant.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_resnet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    def add_common_layers(model):
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        return(model)

    def grouped_convolution_layer_2d(model,
                                     number_of_filters,
                                     strides):
        # Per standard ResNet, this is just a 2-D convolution
        if cardinality == 1:
            grouped_model = Conv2D(filters=number_of_filters,
                                   kernel_size=(3, 3),
                                   strides=strides,
                                   padding='same')(model)
            return(grouped_model)

        if number_of_filters % cardinality != 0:
            raise ValueError('number_of_filters `%` cardinality != 0')

        number_of_group_filters = int(number_of_filters / cardinality)

        convolution_layers = []
        for j in range(cardinality):
            local_layer = Lambda(lambda z: z[:, :, :,
              j * number_of_group_filters:j * number_of_group_filters + number_of_group_filters])(model)
            convolution_layers.append(Conv2D(filters=number_of_group_filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding='same')(local_layer))

        grouped_model = Concatenate()(convolution_layers)
        return(grouped_model)

    def squeeze_and_excite_block_2d(model,
                                    ratio=16):
        initial = model
        number_of_filters = K.int_shape(initial)[-1]

        block_shape = (1, 1, number_of_filters)

        block = GlobalAveragePooling2D()(initial)
        block = Reshape(target_shape=block_shape)(block)
        block = Dense(units=number_of_filters//ratio,
                      activation='relu',
                      kernel_initializer='he_normal',
                      use_bias=False)(block)
        block = Dense(units=number_of_filters,
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=False)(block)

        x = Multiply()([initial, block])
        return(x)

    def residual_block_2d(model,
                         number_of_filters_in,
                         number_of_filters_out,
                         strides=(1, 1),
                         project_shortcut=False,
                         squeeze_and_excite=False):
        shortcut = model

        model = Conv2D(filters=number_of_filters_in,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(model)
        model = add_common_layers(model)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        model = grouped_convolution_layer_2d(model,
                                             number_of_filters=number_of_filters_in,
                                             strides=strides)
        model = add_common_layers(model)

        model = Conv2D(filters=number_of_filters_out,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(model)
        model = BatchNormalization()(model)

        if project_shortcut == True or strides != (1,1):
            shortcut = Conv2D(filters=number_of_filters_out,
                              kernel_size=(1, 1),
                              strides=strides,
                              padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        if squeeze_and_excite == True:
            model = squeeze_and_excite_block_2d(model)

        model = Add()([shortcut, model])
        model = LeakyReLU()(model)
        return(model)


    input_image = Input(shape = input_image_size)

    n_filters = lowest_resolution

    outputs = Conv2D(filters=n_filters,
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     padding='same')(input_image)
    outputs = add_common_layers(outputs)
    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='same')(outputs)

    for i in range(len(layers)):
        n_filters_in = lowest_resolution * 2**layers[i]
        n_filters_out = 2 * n_filters_in

        for j in range(residual_block_schedule[i]):
            project_shortcut = False
            if i == 0 and j == 0:
                project_shortcut = True

            if i > 0 and j == 0:
                strides = (2, 2)
            else:
                strides = (1, 1)

            outputs = residual_block_2d(outputs,
                                        number_of_filters_in=n_filters_in,
                                        number_of_filters_out=n_filters_out,
                                        strides=strides,
                                        project_shortcut=project_shortcut,
                                        squeeze_and_excite=squeeze_and_excite)

    outputs = GlobalAveragePooling2D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layer_activation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    resnet_model = None
    if input_scalars_size > 0:
        input_scalars = Input( shape = (input_scalars_size,) )
        concatenated_layer = Concatenate()([outputs, input_scalars])
        outputs = Dense(units=number_of_classification_labels,
                        activation=layer_activation)(concatenated_layer)
        resnet_model = Model(inputs=[input_image, input_scalars], outputs = outputs)
    else:
        outputs = Dense(units=number_of_classification_labels,
                        activation=layer_activation)(outputs)
        resnet_model = Model(inputs=input_image, outputs=outputs)

    return(resnet_model)

def create_resnet_model_3d(input_image_size,
                           input_scalars_size=0,
                           number_of_classification_labels=1000,
                           layers=(1, 2, 3, 4),
                           residual_block_schedule=(3, 4, 6, 3),
                           lowest_resolution=64,
                           cardinality=1,
                           squeeze_and_excite=False,
                           mode='classification'
                          ):
    """
    3-D implementation of the ResNet deep learning architecture.

    Creates a keras model of the ResNet deep learning architecture for image
    classification.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    input_scalars_size : integer
        Optional integer specifying the size of the input vector for scalars that
        get concatenated to the fully connected layer at the end of the network.

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    residual_block_schedule : tuple
        A tuple defining the how many residual blocks repeats for each layer.

    lowest_resolution : integer
        Number of filters at the initial layer.

    cardinality : integer
        perform ResNet (cardinality = 1) or ResNeX (cardinality does not 1 but,
        instead, powers of 2---try '32').

    squeeze_and_excite : boolean
        add the squeeze-and-excite block variant.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_resnet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    def add_common_layers(model):
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        return(model)

    def grouped_convolution_layer_3d(model,
                                     number_of_filters,
                                     strides):
        # Per standard ResNet, this is just a 3-D convolution
        if cardinality == 1:
            grouped_model = Conv3D(filters=number_of_filters,
                                   kernel_size=(3, 3, 3),
                                   strides=strides,
                                   padding='same')(model)
            return(grouped_model)

        if number_of_filters % cardinality != 0:
            raise ValueError('number_of_filters `%` cardinality != 0')

        number_of_group_filters = int(number_of_filters / cardinality)

        convolution_layers = []
        for j in range(cardinality):
            local_layer = Lambda(lambda z: z[:, :, :, :,
              j * number_of_group_filters:j * number_of_group_filters + number_of_group_filters])(model)
            convolution_layers.append(Conv3D(filters=number_of_group_filters,
                                             kernel_size=(3, 3, 3),
                                             strides=strides,
                                             padding='same')(local_layer))

        grouped_model = Concatenate()(convolution_layers)
        return(grouped_model)

    def squeeze_and_excite_block_3d(model,
                                    ratio=16):
        initial = model
        number_of_filters = K.int_shape(initial)[-1]

        block_shape = (1, 1, 1, number_of_filters)

        block = GlobalAveragePooling3D()(initial)
        block = Reshape(target_shape=block_shape)(block)
        block = Dense(units=number_of_filters//ratio,
                      activation='relu',
                      kernel_initializer='he_normal',
                      use_bias=False)(block)
        block = Dense(units=number_of_filters,
                      activation='sigmoid',
                      kernel_initializer='he_normal',
                      use_bias=False)(block)

        x = Multiply()([initial, block])
        return(x)

    def residual_block_3d(model,
                          number_of_filters_in,
                          number_of_filters_out,
                          strides=(1, 1, 1),
                          project_shortcut=False,
                          squeeze_and_excite=False):
        shortcut = model

        model = Conv3D(filters=number_of_filters_in,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1),
                       padding='same')(model)
        model = add_common_layers(model)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        model = grouped_convolution_layer_3d(model,
                                             number_of_filters=number_of_filters_in,
                                             strides=strides)
        model = add_common_layers(model)

        model = Conv3D(filters=number_of_filters_out,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1),
                       padding='same')(model)
        model = BatchNormalization()(model)

        if project_shortcut == True or strides != (1,1,1):
            shortcut = Conv3D(filters=number_of_filters_out,
                              kernel_size=(1, 1, 1),
                              strides=strides,
                              padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        if squeeze_and_excite == True:
            model = squeeze_and_excite_block_3d(model)

        model = Add()([shortcut, model])
        model = LeakyReLU()(model)
        return(model)


    input_image = Input(shape = input_image_size)

    n_filters = lowest_resolution

    outputs = Conv3D(filters=n_filters,
                     kernel_size=(7, 7, 7),
                     strides=(2, 2, 2),
                     padding='same')(input_image)
    outputs = add_common_layers(outputs)
    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2),
                           padding='same')(outputs)

    for i in range(len(layers)):
        n_filters_in = lowest_resolution * 2**layers[i]
        n_filters_out = 2 * n_filters_in

        for j in range(residual_block_schedule[i]):
            project_shortcut = False
            if i == 0 and j == 0:
                project_shortcut = True

            if i > 0 and j == 0:
                strides = (2, 2, 2)
            else:
                strides = (1, 1, 1)

            outputs = residual_block_3d(outputs,
                                        number_of_filters_in=n_filters_in,
                                        number_of_filters_out=n_filters_out,
                                        strides=strides,
                                        project_shortcut=project_shortcut,
                                        squeeze_and_excite=squeeze_and_excite)


    outputs = GlobalAveragePooling3D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layer_activation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    resnet_model = None
    if input_scalars_size > 0:
        input_scalars = Input( shape = (input_scalars_size,) )
        concatenated_layer = Concatenate()([outputs, input_scalars])
        outputs = Dense(units=number_of_classification_labels,
                        activation=layer_activation)(concatenated_layer)
        resnet_model = Model(inputs=[input_image, input_scalars], outputs = outputs)
    else:
        outputs = Dense(units=number_of_classification_labels,
                        activation=layer_activation)(outputs)
        resnet_model = Model(inputs=input_image, outputs=outputs)

    return(resnet_model)



