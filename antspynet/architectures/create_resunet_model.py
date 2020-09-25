
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dropout, BatchNormalization, Add,
                                     ThresholdedReLU, Concatenate, Dense,
                                     Conv2D, Conv2DTranspose,
                                     MaxPooling2D, UpSampling2D,
                                     Conv3D, Conv3DTranspose,
                                     MaxPooling3D, UpSampling3D)

from tensorflow.keras import initializers
from tensorflow.keras import regularizers

def create_resunet_model_2d(input_image_size,
                            number_of_outputs=1,
                            number_of_filters_at_base_layer=32,
                            bottle_neck_block_depth_schedule=(3, 4),
                            convolution_kernel_size=(3, 3),
                            deconvolution_kernel_size=(2, 2),
                            dropout_rate=0.0,
                            weight_decay=0.0,
                            mode='classification'
                           ):
    """
    2-D implementation of the Resnet + U-net deep learning architecture.

    Creates a keras model of the U-net + ResNet deep learning architecture for
    image segmentation and regression with the paper available here:

            https://arxiv.org/abs/1608.04117

    This particular implementation was ported from the following python
    implementation:

            https://github.com/veugene/fcn_maker/


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

    number_of_filters_at_base_layer : integer
        Number of filters at the beginning and end of the 'U'.  Doubles at each
        descending/ascending layer.

    bottle_neck_block_depth_schedule : tuple
        Tuple that provides the encoding layer schedule for the number of bottleneck
        blocks per long skip connection.

    convolution_kernel_size : tuple of length 2
        2-d vector defining the kernel size during the encoding path

    deconvolution_kernel_size : tuple of length 2
        2-d vector defining the kernel size during the decoding

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay : scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers.  Default = 0.0.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_resunet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    def simple_block_2d(input, number_of_filters,
                        downsample=False, upsample=False,
                        convolution_kernel_size=(3, 3),
                        deconvolution_kernel_size=(2, 2),
                        weight_decay=0.0, dropout_rate=0.0):

        number_of_output_filters = number_of_filters

        output = BatchNormalization()(input)
        output = ThresholdedReLU(theta = 0)(output)

        if downsample:
            output = MaxPooling2D(pool_size=(2, 2))(output)

        output = Conv2D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        if upsample:
            output = Conv2DTranspose(filters=number_of_filters,
                                     kernel_size=deconvolution_kernel_size,
                                     padding='same',
                                     kernel_initializer=initializers.he_normal(),
                                     kernel_regularizer=regularizers.l2(weight_decay))(output)
            output = UpSampling2D(size=(2, 2))(output)

        if dropout_rate > 0.0:
           output=Dropout(rate=dropout_rate)(output)

        # Modify the input so that it has the same size as the output

        if downsample:
            input = Conv2D(filters=number_of_output_filters,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same')(input)
        elif upsample:
            input = Conv2DTranspose(filters=number_of_output_filters,
                                    kernel_size=(1, 1),
                                    padding='same')(input)
            input = UpSampling2D(size=(2, 2))(input)
        elif number_of_filters != number_of_output_filters:
            input = Conv2D(filters=number_of_output_filters,
                            kernel_size=(1, 1),
                            padding='same')(input)

        output = skip_connection(input, output)

        return(output)

    def bottle_neck_block_2d(input, number_of_filters, downsample=False,
                             upsample=False, deconvolution_kernel_size=(2, 2),
                             weight_decay=0.0, dropout_rate=0.0):

        output = input

        number_of_output_filters = number_of_filters

        if downsample:
            output = BatchNormalization()(output)
            output = ThresholdedReLU(theta = 0)(output)

            output = Conv2D(filters=number_of_filters,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            kernel_initializer=initializers.he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay))(output)

        output = BatchNormalization()(output)
        output = ThresholdedReLU(theta = 0)(output)

        output = Conv2D(filters=number_of_filters,
                        kernel_size=(1, 1),
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        output = BatchNormalization()(output)
        output = ThresholdedReLU(theta = 0)(output)

        if upsample:
            output = Conv2DTranspose(filters=number_of_filters,
                                     kernel_size=deconvolution_kernel_size,
                                     padding='same',
                                     kernel_initializer=initializers.he_normal(),
                                     kernel_regularizer=regularizers.l2(weight_decay))(output)
            output = UpSampling2D(size=(2, 2))(output)

        output = Conv2D(filters=(number_of_filters * 4),
                        kernel_size=(1, 1),
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        number_of_output_filters = number_of_filters * 4

        if dropout_rate > 0.0:
           output=Dropout(rate=dropout_rate)(output)

        # Modify the input so that it has the same size as the output

        if downsample:
            input = Conv2D(filters=number_of_output_filters,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same')(input)
        elif upsample:
            input = Conv2DTranspose(filters=number_of_output_filters,
                                    kernel_size=(1, 1),
                                    padding='same')(input)
            input = UpSampling2D(size=(2, 2))(input)
        elif number_of_filters != number_of_output_filters:
            input = Conv2D(filters=number_of_output_filters,
                            kernel_size=(1, 1),
                            padding='valid')(input)

        output = skip_connection(input, output)

        return(output)

    def skip_connection(source, target, merge_mode='sum'):
        layer_list = [source, target]

        output = None
        if merge_mode == 'sum':
            output = Add()(layer_list)
        else:
            channel_axis = 0
            if K.image_data_format() == 'channels_last':
                channel_axis = -1
            output = Concatenate(axis=channel_axis)(layer_list)

        return(output)

    inputs = Input(shape = input_image_size)

    encoding_layers_with_long_skip_connections = []
    encoding_layer_count = 1

    # Preprocessing layer

    model = Conv2D(filters=number_of_filters_at_base_layer,
                   kernel_size=convolution_kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(inputs)

    encoding_layers_with_long_skip_connections.append(model)
    encoding_layer_count += 1

    # Encoding initialization path

    model = simple_block_2d(model, number_of_filters_at_base_layer, downsample=True,
                            convolution_kernel_size=convolution_kernel_size,
                            deconvolution_kernel_size=deconvolution_kernel_size,
                            weight_decay=weight_decay, dropout_rate=dropout_rate)

    encoding_layers_with_long_skip_connections.append(model)
    encoding_layer_count += 1

    # Encoding main path

    number_of_bottle_neck_layers = len(bottle_neck_block_depth_schedule)
    for i in range(number_of_bottle_neck_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**i

        for j in range(bottle_neck_block_depth_schedule[i]):

            do_downsample = False
            if j == 0:
                do_downsample = True
            else:
                do_downsample = False

            model = bottle_neck_block_2d(model, number_of_filters=number_of_filters,
                                         downsample=do_downsample,
                                         deconvolution_kernel_size=deconvolution_kernel_size,
                                         weight_decay=weight_decay, dropout_rate=dropout_rate)

            if j == (bottle_neck_block_depth_schedule[i] - 1):
               encoding_layers_with_long_skip_connections.append(model)
               encoding_layer_count += 1

    encoding_layer_count -= 1

    # Transition path

    number_of_filters = number_of_filters_at_base_layer * 2**number_of_bottle_neck_layers

    model = bottle_neck_block_2d(model, number_of_filters=number_of_filters,
                                 downsample=True,
                                 deconvolution_kernel_size=deconvolution_kernel_size,
                                 weight_decay=weight_decay, dropout_rate=dropout_rate)
    model = bottle_neck_block_2d(model, number_of_filters=number_of_filters,
                                 upsample=True,
                                 deconvolution_kernel_size=deconvolution_kernel_size,
                                 weight_decay=weight_decay, dropout_rate=dropout_rate)

    # Decoding main path

    number_of_bottle_neck_layers = len(bottle_neck_block_depth_schedule)
    for i in range(number_of_bottle_neck_layers):
        number_of_filters = (number_of_filters_at_base_layer *
                             2**(number_of_bottle_neck_layers - i - 1))

        for j in range(bottle_neck_block_depth_schedule[number_of_bottle_neck_layers - i - 1]):

            do_upsample = False
            if j == bottle_neck_block_depth_schedule[number_of_bottle_neck_layers - i - 1] - 1:
                do_upsample = True
            else:
                do_upsample = False

            model = bottle_neck_block_2d(model, number_of_filters=number_of_filters,
                                         upsample=do_upsample,
                                         deconvolution_kernel_size=deconvolution_kernel_size,
                                         weight_decay=weight_decay, dropout_rate=dropout_rate)

            if j == 0:
               model = Conv2D(filters=(number_of_filters * 4),
                              kernel_size=(1, 1),
                              padding='same')(model)
               model = skip_connection(encoding_layers_with_long_skip_connections[encoding_layer_count - 1], model)
               encoding_layer_count -= 1

    # Decoding initialization path

    model = simple_block_2d(model, number_of_filters_at_base_layer, upsample=True,
                            convolution_kernel_size=convolution_kernel_size,
                            deconvolution_kernel_size=deconvolution_kernel_size,
                            weight_decay=weight_decay, dropout_rate=dropout_rate)

    # Postprocessing layer

    model = Conv2D(filters=number_of_filters_at_base_layer,
                   kernel_size=convolution_kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(model)
    encoding_layer_count -= 1

    model = skip_connection(encoding_layers_with_long_skip_connections[encoding_layer_count - 1], model)

    model = BatchNormalization()(model)
    model = ThresholdedReLU(theta = 0)(model)

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
                     kernel_regularizer=regularizers.l2(weight_decay))(model)

    resunet_model = Model(inputs=inputs, outputs=outputs)

    return resunet_model

def create_resunet_model_3d(input_image_size,
                            number_of_outputs=1,
                            number_of_filters_at_base_layer=32,
                            bottle_neck_block_depth_schedule=(3, 4),
                            convolution_kernel_size=(3, 3, 3),
                            deconvolution_kernel_size=(2, 2, 2),
                            dropout_rate=0.0,
                            weight_decay=0.0,
                            mode='classification'
                           ):
    """
    3-D implementation of the Resnet + U-net deep learning architecture.

    Creates a keras model of the U-net + ResNet deep learning architecture for
    image segmentation and regression with the paper available here:

            https://arxiv.org/abs/1608.04117

    This particular implementation was ported from the following python
    implementation:

            https://github.com/veugene/fcn_maker/


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

    number_of_filters_at_base_layer : integer
        Number of filters at the beginning and end of the 'U'.  Doubles at each
        descending/ascending layer.

    bottle_neck_block_depth_schedule : tuple
        Tuple that provides the encoding layer schedule for the number of bottleneck
        blocks per long skip connection.

    convolution_kernel_size : tuple of length 3
        3-d vector defining the kernel size during the encoding path

    deconvolution_kernel_size : tuple of length 3
        3-d vector defining the kernel size during the decoding

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay : scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers.  Default = 0.0.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_resunet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    def simple_block_3d(input, number_of_filters,
                        downsample=False, upsample=False,
                        convolution_kernel_size=(3, 3, 3),
                        deconvolution_kernel_size=(2, 2, 2),
                        weight_decay=0.0, dropout_rate=0.0):

        number_of_output_filters = number_of_filters

        output = BatchNormalization()(input)
        output = ThresholdedReLU(theta = 0)(output)

        if downsample:
            output = MaxPooling3D(pool_size=(2, 2, 2))(output)

        output = Conv3D(filters=number_of_filters,
                        kernel_size=convolution_kernel_size,
                        padding='same',
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        if upsample:
            output = Conv3DTranspose(filters=number_of_filters,
                                     kernel_size=deconvolution_kernel_size,
                                     padding='same',
                                     kernel_initializer=initializers.he_normal(),
                                     kernel_regularizer=regularizers.l2(weight_decay))(output)
            output = UpSampling3D(size=(2, 2, 2))(output)

        if dropout_rate > 0.0:
           output=Dropout(rate=dropout_rate)(output)

        # Modify the input so that it has the same size as the output

        if downsample:
            input = Conv3D(filters=number_of_output_filters,
                            kernel_size=(1, 1, 1),
                            strides=(2, 2, 2),
                            padding='same')(input)
        elif upsample:
            input = Conv3DTranspose(filters=number_of_output_filters,
                                    kernel_size=(1, 1, 1),
                                    padding='same')(input)
            input = UpSampling3D(size=(2, 2, 2))(input)
        elif number_of_filters != number_of_output_filters:
            input = Conv3D(filters=number_of_output_filters,
                            kernel_size=(1, 1, 1),
                            padding='same')(input)

        output = skip_connection(input, output)

        return(output)

    def bottle_neck_block_3d(input, number_of_filters, downsample=False,
                             upsample=False, deconvolution_kernel_size=(2, 2, 2),
                             weight_decay=0.0, dropout_rate=0.0):

        output = input

        number_of_output_filters = number_of_filters

        if downsample:
            output = BatchNormalization()(output)
            output = ThresholdedReLU(theta = 0)(output)

            output = Conv3D(filters=number_of_filters,
                            kernel_size=(1, 1, 1),
                            strides=(2, 2, 2),
                            kernel_initializer=initializers.he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay))(output)

        output = BatchNormalization()(output)
        output = ThresholdedReLU(theta = 0)(output)

        output = Conv3D(filters=number_of_filters,
                        kernel_size=(1, 1, 1),
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        output = BatchNormalization()(output)
        output = ThresholdedReLU(theta = 0)(output)

        if upsample:
            output = Conv3DTranspose(filters=number_of_filters,
                                     kernel_size=deconvolution_kernel_size,
                                     padding='same',
                                     kernel_initializer=initializers.he_normal(),
                                     kernel_regularizer=regularizers.l2(weight_decay))(output)
            output = UpSampling3D(size=(2, 2, 2))(output)

        output = Conv3D(filters=(number_of_filters * 4),
                        kernel_size=(1, 1, 1),
                        kernel_initializer=initializers.he_normal(),
                        kernel_regularizer=regularizers.l2(weight_decay))(output)

        number_of_output_filters = number_of_filters * 4

        if dropout_rate > 0.0:
           output=Dropout(rate=dropout_rate)(output)

        # Modify the input so that it has the same size as the output

        if downsample:
            input = Conv3D(filters=number_of_output_filters,
                            kernel_size=(1, 1, 1),
                            strides=(2, 2, 2),
                            padding='same')(input)
        elif upsample:
            input = Conv3DTranspose(filters=number_of_output_filters,
                                    kernel_size=(1, 1, 1),
                                    padding='same')(input)
            input = UpSampling3D(size=(2, 2, 2))(input)
        elif number_of_filters != number_of_output_filters:
            input = Conv3D(filters=number_of_output_filters,
                            kernel_size=(1, 1, 1),
                            padding='valid')(input)

        output = skip_connection(input, output)

        return(output)

    def skip_connection(source, target, merge_mode='sum'):
        layer_list = [source, target]

        output = None
        if merge_mode == 'sum':
            output = Add()(layer_list)
        else:
            channel_axis = 0
            if K.image_data_format() == 'channels_last':
                channel_axis = -1
            output = Concatenate(axis=channel_axis)(layer_list)

        return(output)

    inputs = Input(shape = input_image_size)

    encoding_layers_with_long_skip_connections = []
    encoding_layer_count = 1

    # Preprocessing layer

    model = Conv3D(filters=number_of_filters_at_base_layer,
                   kernel_size=convolution_kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(inputs)

    encoding_layers_with_long_skip_connections.append(model)
    encoding_layer_count += 1

    # Encoding initialization path

    model = simple_block_3d(model, number_of_filters_at_base_layer, downsample=True,
                            convolution_kernel_size=convolution_kernel_size,
                            deconvolution_kernel_size=deconvolution_kernel_size,
                            weight_decay=weight_decay, dropout_rate=dropout_rate)

    encoding_layers_with_long_skip_connections.append(model)
    encoding_layer_count += 1

    # Encoding main path

    number_of_bottle_neck_layers = len(bottle_neck_block_depth_schedule)
    for i in range(number_of_bottle_neck_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**i

        for j in range(bottle_neck_block_depth_schedule[i]):

            do_downsample = False
            if j == 0:
                do_downsample = True
            else:
                do_downsample = False

            model = bottle_neck_block_3d(model, number_of_filters=number_of_filters,
                                         downsample=do_downsample,
                                         deconvolution_kernel_size=deconvolution_kernel_size,
                                         weight_decay=weight_decay, dropout_rate=dropout_rate)

            if j == (bottle_neck_block_depth_schedule[i] - 1):
               encoding_layers_with_long_skip_connections.append(model)
               encoding_layer_count += 1

    encoding_layer_count -= 1

    # Transition path

    number_of_filters = number_of_filters_at_base_layer * 2**number_of_bottle_neck_layers

    model = bottle_neck_block_3d(model, number_of_filters=number_of_filters,
                                 downsample=True,
                                 deconvolution_kernel_size=deconvolution_kernel_size,
                                 weight_decay=weight_decay, dropout_rate=dropout_rate)
    model = bottle_neck_block_3d(model, number_of_filters=number_of_filters,
                                 upsample=True,
                                 deconvolution_kernel_size=deconvolution_kernel_size,
                                 weight_decay=weight_decay, dropout_rate=dropout_rate)

    # Decoding main path

    number_of_bottle_neck_layers = len(bottle_neck_block_depth_schedule)
    for i in range(number_of_bottle_neck_layers):
        number_of_filters = (number_of_filters_at_base_layer *
                             2**(number_of_bottle_neck_layers - i - 1))

        for j in range(bottle_neck_block_depth_schedule[number_of_bottle_neck_layers - i - 1]):

            do_upsample = False
            if j == bottle_neck_block_depth_schedule[number_of_bottle_neck_layers - i - 1] - 1:
                do_upsample = True
            else:
                do_upsample = False

            model = bottle_neck_block_3d(model, number_of_filters=number_of_filters,
                                         upsample=do_upsample,
                                         deconvolution_kernel_size=deconvolution_kernel_size,
                                         weight_decay=weight_decay, dropout_rate=dropout_rate)

            if j == 0:
               model = Conv3D(filters=(number_of_filters * 4),
                              kernel_size=(1, 1, 1),
                              padding='same')(model)
               model = skip_connection(encoding_layers_with_long_skip_connections[encoding_layer_count - 1], model)
               encoding_layer_count -= 1

    # Decoding initialization path

    model = simple_block_3d(model, number_of_filters_at_base_layer, upsample=True,
                            convolution_kernel_size=convolution_kernel_size,
                            deconvolution_kernel_size=deconvolution_kernel_size,
                            weight_decay=weight_decay, dropout_rate=dropout_rate)

    # Postprocessing layer

    model = Conv3D(filters=number_of_filters_at_base_layer,
                   kernel_size=convolution_kernel_size,
                   activation='relu',
                   padding='same',
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay))(model)
    encoding_layer_count -= 1

    model = skip_connection(encoding_layers_with_long_skip_connections[encoding_layer_count - 1], model)

    model = BatchNormalization()(model)
    model = ThresholdedReLU(theta = 0)(model)

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
                     kernel_regularizer=regularizers.l2(weight_decay))(model)

    resunet_model = Model(inputs=inputs, outputs=outputs)

    return resunet_model
