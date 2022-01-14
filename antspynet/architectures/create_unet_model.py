from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Concatenate, Dense, Dropout, Add, Activation,
                                     multiply, ReLU, LeakyReLU,
                                     GlobalAveragePooling2D,
                                     Conv2D, Conv2DTranspose, MaxPooling2D,
                                     UpSampling2D,
                                     GlobalAveragePooling3D,
                                     Conv3D, Conv3DTranspose, MaxPooling3D,
                                     UpSampling3D)
from tensorflow.keras import regularizers

from ..utilities import InstanceNormalization

def create_unet_model_2d(input_image_size,
                         number_of_outputs=2,
                         scalar_output_size = 0,
                         scalar_output_activation = "relu",
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         number_of_filters=None,
                         convolution_kernel_size=(3, 3),
                         deconvolution_kernel_size=(2, 2),
                         pool_size=(2, 2),
                         strides=(2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         mode='classification',
                         additional_options=None
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

    scalar_output_size : integer
        If greater than 0, a global average pooling from each
        encoding layer is concatenated to a dense layer as a secondary output.

    scalar_output_activation : string
        Activation for nonzero output scalar.

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

    mode :  string
        `classification`, `regression`, or `sigmoid`.  Default = `classification`.

    additional_options : string or tuple of strings
        specific configuration add-ons/tweaks:
            * "attentionGating" -- attention-unet variant in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "nnUnetActivationStyle" -- U-net activation explained in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "initialConvolutionalKernelSize[X]" -- Set the first two convolutional layer kernel sizes to X.

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

    # Handle additional specific configurations

    initial_convolution_kernel_size = convolution_kernel_size
    add_attention_gating = False
    nn_unet_activation_style = False

    if additional_options is not None:

        if "attentionGating" in additional_options:
            add_attention_gating = True

        if "nnUnetActivationStyle" in additional_options:
            nn_unet_activation_style = True

        option = [o for o in additional_options if o.startswith('initialConvolutionKernelSize')]
        if not not option:
            initial_convolution_kernel_size = option[0].replace("initialConvolutionKernelSize", "")
            initial_convolution_kernel_size = initial_convolution_kernel_size.replace("[", "")
            initial_convolution_kernel_size = int(initial_convolution_kernel_size.replace("]", ""))

    # Specify the number of filters

    if number_of_filters is not None:
        number_of_layers = len(number_of_filters)
    else:
        number_of_filters = list()
        for i in range(number_of_layers):
            number_of_filters.append(number_of_filters_at_base_layer * 2**i)

    inputs = Input(shape = input_image_size)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):

        if i == 0:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=initial_convolution_kernel_size,
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

        if i == 0:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=initial_convolution_kernel_size,
                          padding='same')(conv)
        else:
            conv = Conv2D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same')(conv)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling2D(pool_size=pool_size,
                                strides=strides)(encoding_convolution_layers[i])

    scalar_output = None
    if scalar_output_size > 0:
        scalar_layers = list()
        for i in range(number_of_layers):
            scalar_layers.append(GlobalAveragePooling2D()(encoding_convolution_layers[i]))
        scalar_output = Concatenate()(scalar_layers)
        scalar_output = Dense(units=scalar_output_size,
                              activation=scalar_output_activation)(scalar_output)

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

    conv_activation = ''

    if mode == 'sigmoid':
        conv_activation = 'sigmoid'
    elif mode == 'classification':
        conv_activation = 'softmax'
    elif mode == 'regression':
        conv_activation = 'linear'
    else:
        raise ValueError('mode must be either `classification`, `regression` or `sigmoid`.')

    outputs = Conv2D(filters=number_of_outputs,
                     kernel_size=(1, 1),
                     activation = conv_activation,
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = None
    if scalar_output_size > 0:
        unet_model = Model(inputs=inputs, outputs=[outputs, scalar_output])
    else:
        unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


def create_unet_model_3d(input_image_size,
                         number_of_outputs=2,
                         scalar_output_size = 0,
                         scalar_output_activation = "relu",
                         number_of_layers=4,
                         number_of_filters_at_base_layer=32,
                         number_of_filters=None,
                         convolution_kernel_size=(3, 3, 3),
                         deconvolution_kernel_size=(2, 2, 2),
                         pool_size=(2, 2, 2),
                         strides=(2, 2, 2),
                         dropout_rate=0.0,
                         weight_decay=0.0,
                         mode='classification',
                         additional_options=None
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

    scalar_output_size : integer
        If greater than 0, a global average pooling from each
        encoding layer is concatenated to a dense layer as a secondary output.

    scalar_output_activation : string
        Activation for nonzero output scalar.

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

    mode :  string
        `classification` `regression`, or `sigmoid`.  Default = `classification`.

    additional_options : string or tuple of strings
        specific configuration add-ons/tweaks:
            * "attentionGating" -- attention-unet variant in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "nnUnetActivationStyle" -- U-net activation explained in https://pubmed.ncbi.nlm.nih.gov/33288961/
            * "initialConvolutionalKernelSize[X]" -- Set the first two convolutional layer kernel sizes to X.

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

    # Handle additional specific configurations

    initial_convolution_kernel_size = convolution_kernel_size
    add_attention_gating = False
    nn_unet_activation_style = False

    if additional_options is not None:

        if "attentionGating" in additional_options:
            add_attention_gating = True

        if "nnUnetActivationStyle" in additional_options:
            nn_unet_activation_style = True

        option = [o for o in additional_options if o.startswith('initialConvolutionKernelSize')]
        if not not option:
            initial_convolution_kernel_size = option[0].replace("initialConvolutionKernelSize", "")
            initial_convolution_kernel_size = initial_convolution_kernel_size.replace("[", "")
            initial_convolution_kernel_size = int(initial_convolution_kernel_size.replace("]", ""))

    # Specify the number of filters

    if number_of_filters is not None:
        number_of_layers = len(number_of_filters)
    else:
        number_of_filters = list()
        for i in range(number_of_layers):
            number_of_filters.append(number_of_filters_at_base_layer * 2**i)

    inputs = Input(shape = input_image_size)

    # Encoding path

    encoding_convolution_layers = []
    pool = None
    for i in range(number_of_layers):

        if i == 0:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=initial_convolution_kernel_size,
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

        if i == 0:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=initial_convolution_kernel_size,
                          padding='same')(conv)
        else:
            conv = Conv3D(filters=number_of_filters[i],
                          kernel_size=convolution_kernel_size,
                          padding='same')(conv)

        if nn_unet_activation_style == True:
            conv = nn_unet_activation(conv)
        else:
            conv = ReLU()(conv)

        encoding_convolution_layers.append(conv)

        if i < number_of_layers - 1:
            pool = MaxPooling3D(pool_size=pool_size,
                                strides=strides)(encoding_convolution_layers[i])

    scalar_output = None
    if scalar_output_size > 0:
        scalar_layers = list()
        for i in range(number_of_layers):
            scalar_layers.append(GlobalAveragePooling3D()(encoding_convolution_layers[i]))
        scalar_output = Concatenate()(scalar_layers)
        scalar_output = Dense(units=scalar_output_size,
                              activation=scalar_output_activation)(scalar_output)

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

    conv_activation = ''
    if mode == 'sigmoid':
        conv_activation = 'sigmoid'
    elif mode == 'classification':
        conv_activation = 'softmax'
    elif mode == 'regression':
        conv_activation = 'linear'
    else:
        raise ValueError('mode must be either `classification`, `regression` or `sigmoid`.')

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=(1, 1, 1),
                     activation=conv_activation,
                     kernel_regularizer=regularizers.l2(weight_decay))(outputs)

    unet_model = None
    if scalar_output_size > 0:
        unet_model = Model(inputs=inputs, outputs=[outputs, scalar_output])
    else:
        unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model
