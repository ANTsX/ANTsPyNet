from keras.models import Model
from keras.layers import (Input, Conv2D, Conv2DTranspose,
                          MaxPooling2D, Concatenate, UpSampling2D,
                          Conv3D, Conv3DTranspose, MaxPooling3D,
                          UpSampling3D, Dropout)
from keras import regularizers

def create_unet_model_2d(input_image_size,
                         number_of_outputs=1,
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         convolution_kernel_size=(3, 3),
                         deconvolution_kernel_size=(2, 2),
                         pool_size=(2, 2),
                         strides=(2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         mode='classification'
                        ):
    """
    2-D implementation of the U-net deep learning architecture.

    Creates a keras model of the U-net deep learning architecture for image
    segmentation and regression.  More information is provided at the authors'
    website:

            https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

    with the paper available here:

            https://arxiv.org/abs/1505.04597

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/joelthelion/ultrasound-nerve-segmentation

    :param input_image_size: Used for specifying the input tensor shape.  The
      shape (or dimension) of that tensor is the image dimensions followed by
      the number of channels (e.g., red, green, and blue).  The batch size
      (i.e., number of training images) is not specified a priori.
    :param number_of_outputs Meaning depends on the mode.  For
      `classification` this is the number of segmentation labels.  For
      `regression` this is the number of outputs.
    :param number_of_layers: number of encoding/decoding layers.
    :param number_of_filters_at_base_layer: number of filters at the beginning
      and end of the `U`.  Doubles at each descending/ascending layer.
    :param convolution_kernel_size: 2-d array defining the kernel size
      during the encoding path.
    :param deconvolution_kernel_size: 2-d array defining the kernel size
      during the decoding.
    :param pool_size: 2-d array defining the region for each pooling layer.
    :param strides: 2-d array describing the stride length in each direction.
    :param dropout_rate: float between 0 and 1 to use between dense layers.
    :param weight_decay weighting parameter for L2 regularization of the
      kernel weights of the convolution layers.  Default = 0.0.
    :param mode `classification` or `regression`.  Default = `classification`.
    :returns: a u-net keras model.
    :raises: ValueError: raises an exception if `mode` is incorrect.

    """

    inputs = Input(shape = input_image_size)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**i

        if i == 0:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        else:
            conv = Conv2D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(pool)

        if dropout_rate > 0.0:
            conv = Dropout(rate=dropout_rate)(conv)

        encoding_convolution_layers.append(Conv2D(filters=number_of_filters,
                                                  kernel_size=convolution_kernel_size,
                                                  activation='relu',
                                                  padding='same')(conv))

        if i < number_of_layers - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**(number_of_layers - i - 1)
        deconv = Conv2DTranspose(filters=number_of_filters,
                                 kernel_size=deconvolution_kernel_size,
                                 padding='same',
                                 kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        deconv = UpSampling2D(size=pool_size)(deconv)
        outputs = Concatenate(axis=3)([deconv, encoding_convolution_layers[number_of_layers-i-1]])

        outputs = Conv2D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

        if dropout_rate > 0.0:
            outputs = Dropout(rate=dropout_rate)(outputs)

        outputs = Conv2D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    convActivation = ''

    if mode == 'classification':
        if number_of_outputs == 2:
            convActivation = 'sigmoid'
        else:
            convActivation = 'softmax'
    elif mode == 'regression':
        convActivation = 'linear'
    else:
        raise ValueError( 'mode must be either `classification` or `regression`.')

    outputs = Conv2D(filters=number_of_outputs,
                     kernel_size=(1, 1),
                     activation = convActivation,
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


def create_unet_model_3d(input_image_size,
                         number_of_outputs=1,
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         convolution_kernel_size=(3, 3, 3),
                         deconvolution_kernel_size=(2, 2, 2),
                         pool_size=(2, 2, 2),
                         strides=(2, 2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         mode='classification'
                        ):
    """
    3-D implementation of the U-net deep learning architecture.

    Creates a keras model of the U-net deep learning architecture for image
    segmentation and regression.  More information is provided at the authors'
    website:

            https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

    with the paper available here:

            https://arxiv.org/abs/1505.04597

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/joelthelion/ultrasound-nerve-segmentation

    :param input_image_size: Used for specifying the input tensor shape.  The
      shape (or dimension) of that tensor is the image dimensions followed by
      the number of channels (e.g., red, green, and blue).  The batch size
      (i.e., number of training images) is not specified a priori.
    :param number_of_outputs Meaning depends on the mode.  For
      `classification` this is the number of segmentation labels.  For
      `regression` this is the number of outputs.
    :param number_of_layers: number of encoding/decoding layers.
    :param number_of_filters_at_base_layer: number of filters at the beginning
      and end of the `U`.  Doubles at each descending/ascending layer.
    :param convolution_kernel_size: 3-d array defining the kernel size
      during the encoding path.
    :param deconvolution_kernel_size: 3-d array defining the kernel size
      during the decoding.
    :param pool_size: 3-d array defining the region for each pooling layer.
    :param strides: 3-d array describing the stride length in each direction.
    :param dropout_rate: float between 0 and 1 to use between dense layers.
    :param weight_decay weighting parameter for L2 regularization of the
      kernel weights of the convolution layers.  Default = 0.0.
    :param mode `classification` or `regression`.  Default = `classification`.
    :returns: a u-net keras model.
    :raises: ValueError: raises an exception if `mode` is incorrect.

    """

    inputs = Input(shape = input_image_size)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**i

        if i == 0:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        else:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          activation='relu',
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(pool)

        if dropout_rate > 0.0:
            conv = Dropout(rate=dropout_rate)(conv)

        encoding_convolution_layers.append(Conv3D(filters=number_of_filters,
                                                  kernel_size=convolution_kernel_size,
                                                  activation='relu',
                                                  padding='same')(conv))

        if i < number_of_layers - 1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2**(number_of_layers - i - 1)
        deconv = Conv3DTranspose(filters=number_of_filters,
                                 kernel_size=deconvolution_kernel_size,
                                 padding='same',
                                 kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        deconv = UpSampling3D(size=pool_size)(deconv)
        outputs = Concatenate(axis=3)([deconv, encoding_convolution_layers[number_of_layers-i-1]])

        outputs = Conv3D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

        if dropout_rate > 0.0:
            outputs = Dropout(rate=dropout_rate)(outputs)

        outputs = Conv3D(filters=number_of_filters,
                         kernel_size=convolution_kernel_size,
                         activation='relu',
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    convActivation = ''

    if mode == 'classification':
        if number_of_outputs == 2:
            convActivation = 'sigmoid'
        else:
            convActivation = 'softmax'
    elif mode == 'regression':
        convActivation = 'linear'
    else:
        raise ValueError( 'mode must be either `classification` or `regression`.')

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=(1, 1, 1),
                     activation = convActivation,
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


