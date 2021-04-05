from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Concatenate, Dropout, Add, Activation,
                                     multiply, ReLU, LeakyReLU,
                                     Conv2D, Conv2DTranspose, MaxPooling2D,
                                     UpSampling2D,
                                     Conv3D, Conv3DTranspose, MaxPooling3D,
                                     UpSampling3D)
from tensorflow.keras import regularizers

from ..utilities import InstanceNormalization

def create_unet_model_2d(input_image_size,
                         number_of_outputs=2,
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         number_of_filters=None,
                         convolution_kernel_size=(3, 3),
                         deconvolution_kernel_size=(2, 2),
                         pool_size=(2, 2),
                         strides=(2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         nn_unet_activation_style=False,
                         add_attention_gating=False,
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


    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_outputs : integer
        Meaning depends on the mode.  For `classification` this is the number of
        segmentation labels.  For `regression` this is the number of outputs.

    number_of_layers : integer
        number of encoding/decoding layers.

    number_of_filters_at_base_layer : integer
        number of filters at the beginning and end of the `U`.  Doubles at each
        descending/ascending layer.

    number_of_filters : tuple
        tuple explicitly setting the number of filters at each layer.  One can
        either set this or number_of_layers and  number_of_filters_at_base_layer.
        Default = None.

    convolution_kernel_size : tuple of length 2
        Defines the kernel size during the encoding.

    deconvolution_kernel_size : tuple of length 2
        Defines the kernel size during the decoding.

    pool_size : tuple of length 2
        Defines the region for each pooling layer.

    strides : tuple of length 2
        Strides for the convolutional layers.

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers.  Default = 0.0.

    nn_unet_activation_style : boolean
        Instance normalization followed by leaky relu Activation on the convolution
        layers.

    add_attention_gating :  boolean
        Whether or not to include attention gating.

    mode :  string
        `classification` or `regression`.  Default = `classification`.

    Returns
    -------
    Keras model
        A 2-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_unet_model_2d((128, 128, 1))
    >>> model.summary()
    """

    def nn_unet_activation(x):
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        return x

    def attention_gate_2d(x, g, inter_shape):
        x_theta = Conv2D(filters=inter_shape,
                         kernel_size=(1, 1),
                         strides=(1, 1))(x)
        g_phi = Conv2D(filters=inter_shape,
                       kernel_size=(1, 1),
                       strides=(1, 1))(g)
        f = Add()([x_theta, g_phi])
        f = ReLU()(f)
        f_psi = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=(1, 1))(f)
        alpha = Activation('sigmoid')(f_psi)
        attention = multiply([x, alpha])
        return attention

    inputs = Input(shape = input_image_size)

    if number_of_filters is not None:
        number_of_layers = len(number_of_filters)
    else:
        number_of_filters = list()
        for i in range(number_of_layers):
            number_of_filters.append(number_of_filters_at_base_layer * 2**i)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):

        if i == 0:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        else:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(pool)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        if dropout_rate > 0.0:
            conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv2D(filters=number_of_filters[i],
                      kernel_size=convolution_kernel_size,
                      padding='same')(conv)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling2D(pool_size=pool_size)(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        deconv = Conv2DTranspose(filters=number_of_filters[number_of_layers-i-1],
                                 kernel_size=deconvolution_kernel_size,
                                 padding='same',
                                 kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        if nn_unet_activation_style == True:
            deconv = nn_unet_activation(deconv)
        deconv = UpSampling2D(size=pool_size)(deconv)

        if add_attention_gating == True:
            outputs = attention_gate_2d(deconv,
              encoding_convolution_layers[number_of_layers-i-1],
              number_of_filters[number_of_layers-i-1] // 4)
            outputs = Concatenate(axis=3)([deconv, outputs])
        else:
            outputs = Concatenate(axis=3)([deconv, encoding_convolution_layers[number_of_layers-i-1]])

        outputs = Conv2D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        if nn_unet_activation_style == True:
            outputs = nn_unet_activation(outputs)
        else:
            outputs = ReLU()(outputs)

        if dropout_rate > 0.0:
            outputs = Dropout(rate=dropout_rate)(outputs)

        outputs = Conv2D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

        if nn_unet_activation_style == True:
            outputs = nn_unet_activation(outputs)
        else:
            outputs = ReLU()(outputs)

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
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


def create_unet_model_3d(input_image_size,
                         number_of_outputs=2,
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         number_of_filters=None,
                         convolution_kernel_size=(3, 3, 3),
                         deconvolution_kernel_size=(2, 2, 2),
                         pool_size=(2, 2, 2),
                         strides=(2, 2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         nn_unet_activation_style=False,
                         add_attention_gating=False,
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


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_outputs : integer
        Meaning depends on the mode.  For `classification` this is the number of
        segmentation labels.  For `regression` this is the number of outputs.

    number_of_layers : integer
        number of encoding/decoding layers.

    number_of_filters_at_base_layer : integer
        number of filters at the beginning and end of the `U`.  Doubles at each
        descending/ascending layer.

    number_of_filters : tuple
        tuple explicitly setting the number of filters at each layer.  One can
        either set this or number_of_layers and  number_of_filters_at_base_layer.
        Default = None.

    convolution_kernel_size : tuple of length 3
        Defines the kernel size during the encoding.

    deconvolution_kernel_size : tuple of length 3
        Defines the kernel size during the decoding.

    pool_size : tuple of length 3
        Defines the region for each pooling layer.

    strides : tuple of length 3
        Strides for the convolutional layers.

    dropout_rate : scalar
        Float between 0 and 1 to use between dense layers.

    weight_decay :  scalar
        Weighting parameter for L2 regularization of the kernel weights of the
        convolution layers.  Default = 0.0.

    nn_unet_activation_style : boolean
        Instance normalization followed by leaky relu Activation on the convolution
        layers.

    add_attention_gating :  boolean
        Whether or not to include attention gating.

    mode :  string
        `classification` or `regression`.  Default = `classification`.

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> model = create_unet_model_3d((128, 128, 128, 1))
    >>> model.summary()
    """

    def nn_unet_activation(x):
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        return x

    def attention_gate_3d(x, g, inter_shape):
        x_theta = Conv3D(filters=inter_shape,
                         kernel_size=(1, 1, 1),
                         strides=(1, 1, 1))(x)
        g_phi = Conv3D(filters=inter_shape,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1))(g)
        f = Add()([x_theta, g_phi])
        f = ReLU()(f)
        f_psi = Conv3D(filters=1,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1))(f)
        alpha = Activation('sigmoid')(f_psi)
        attention = multiply([x, alpha])
        return attention

    inputs = Input(shape = input_image_size)

    if number_of_filters is not None:
        number_of_layers = len(number_of_filters)
    else:
        number_of_filters = list()
        for i in range(number_of_layers):
            number_of_filters.append(number_of_filters_at_base_layer * 2**i)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):

        if i == 0:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        else:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same',
                          kernel_regularizer=regularizers.l2(weight_decay))(pool)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        if dropout_rate > 0.0:
            conv = Dropout(rate=dropout_rate)(conv)

        conv = Conv3D(filters=number_of_filters[i],
                      kernel_size=convolution_kernel_size,
                      padding='same')(conv)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers - 1]
    for i in range(1, number_of_layers):
        deconv = Conv3DTranspose(filters=number_of_filters[number_of_layers-i-1],
                                 kernel_size=deconvolution_kernel_size,
                                 padding='same',
                                 kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        if nn_unet_activation_style == True:
            deconv = nn_unet_activation(deconv)
        deconv = UpSampling3D(size=pool_size)(deconv)

        if add_attention_gating == True:
            outputs = attention_gate_3d(deconv,
              encoding_convolution_layers[number_of_layers-i-1],
              number_of_filters[number_of_layers-i-1] // 4)
            outputs = Concatenate(axis=4)([deconv, outputs])
        else:
            outputs = Concatenate(axis=4)([deconv, encoding_convolution_layers[number_of_layers-i-1]])

        outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)
        if nn_unet_activation_style == True:
            outputs = nn_unet_activation(outputs)
        else:
            outputs = ReLU()(outputs)

        if dropout_rate > 0.0:
            outputs = Dropout(rate=dropout_rate)(outputs)

        outputs = Conv3D(filters=number_of_filters[number_of_layers-i-1],
                         kernel_size=convolution_kernel_size,
                         padding='same',
                         kernel_regularizer=regularizers.l2(weight_decay))(outputs)

        if nn_unet_activation_style == True:
            outputs = nn_unet_activation(outputs)
        else:
            outputs = ReLU()(outputs)

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
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model
