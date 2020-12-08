
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Flatten, Dense,
                          Conv2D, Conv2DTranspose, MaxPooling2D,
                          ZeroPadding2D,
                          Conv3D, Conv3DTranspose, MaxPooling3D,
                          ZeroPadding3D)

def create_vgg_model_2d(input_image_size,
                        number_of_classification_labels=1000,
                        layers=(1, 2, 3, 4, 4),
                        lowest_resolution=64,
                        convolution_kernel_size=(3, 3),
                        pool_size=(2, 2),
                        strides=(2, 2),
                        number_of_dense_units=4096,
                        dropout_rate=0.0,
                        style=19,
                        mode='classification'):
    """
    2-D implementation of the Vgg deep learning architecture.

    Creates a keras model of the Vgg deep learning architecture for image
    recognition based on the paper

    K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for
    Large-Scale Image Recognition

    available here:

            https://arxiv.org/abs/1409.1556

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

    Arguments
    ---------

    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    lowest_resolution : integer
        Number of filters at the beginning.

    convolution_kernel_size : tuple
        2-d vector definining the kernel size during the encoding path

    pool_size : tuple
        2-d vector defining the region for each pooling layer.

    strides : tuple
        2-d vector describing the stride length in each direction.

    number_of_dense_units : integer
        Number of units in the last layers.

    dropout_rate : scalar
       Between 0 and 1 to use between dense layers.

    style : integer
       '16' or '19' for VGG16 or VGG19, respectively.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_vgg_model_2d((128, 128, 1))
    >>> model.summary()

    """

    if style != 16 and style != 19:
        raise ValueError("Incorrect style.  Must be either '16' or '19'.")

    inputs = Input(shape = input_image_size)
    outputs = None

    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i] - 1)

        if i == 0:
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(inputs)
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
        elif i == 1:
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
        else:
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv2D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            if style == 19:
                outputs = Conv2D(filters=number_of_filters,
                                 kernel_size=convolution_kernel_size,
                                 activation='relu',
                                 padding='same')(outputs)
        outputs = MaxPooling2D(pool_size=pool_size,
                               strides=strides)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(units=number_of_dense_units,
                    activation ='relu')(outputs)
    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)
    outputs = Dense(units=number_of_dense_units,
                    activation ='relu')(outputs)
    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('unrecognized mode.')

    outputs = Dense(units=number_of_classification_labels,
                    activation =layer_activation)(outputs)

    vgg_model = Model(inputs=inputs, outputs=outputs)

    return(vgg_model)

def create_vgg_model_3d(input_image_size,
                        number_of_classification_labels=1000,
                        layers=(1, 2, 3, 4, 4),
                        lowest_resolution=64,
                        convolution_kernel_size=(3, 3, 3),
                        pool_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        number_of_dense_units=4096,
                        dropout_rate=0.0,
                        style=19,
                        mode='classification'):
    """
    3-D implementation of the Vgg deep learning architecture.

    Creates a keras model of the Vgg deep learning architecture for image
    recognition based on the paper

    K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for
    Large-Scale Image Recognition

    available here:

            https://arxiv.org/abs/1409.1556

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

    Arguments
    ---------

    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : integer
        Number of classification labels.

    layers : tuple
        A tuple determining the number of 'filters' defined at for each layer.

    lowest_resolution : integer
        Number of filters at the beginning.

    convolution_kernel_size : tuple
        3-d vector definining the kernel size during the encoding path

    pool_size : tuple
        3-d vector defining the region for each pooling layer.

    strides : tuple
        3-d vector describing the stride length in each direction.

    number_of_dense_units : integer
        Number of units in the last layers.

    dropout_rate : scalar
       Between 0 and 1 to use between dense layers.

    style : integer
       '16' or '19' for VGG16 or VGG19, respectively.

    mode : string
        'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_vgg_model_3d((128, 128, 128, 1))
    >>> model.summary()

    """

    if style != 16 and style != 19:
        raise ValueError("Incorrect style.  Must be either '16' or '19'.")

    inputs = Input(shape = input_image_size)
    outputs = None

    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i] - 1)

        if i == 0:
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(inputs)
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
        elif i == 1:
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
        else:
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            outputs = Conv3D(filters=number_of_filters,
                             kernel_size=convolution_kernel_size,
                             activation='relu',
                             padding='same')(outputs)
            if style == 19:
                outputs = Conv3D(filters=number_of_filters,
                                 kernel_size=convolution_kernel_size,
                                 activation='relu',
                                 padding='same')(outputs)
        outputs = MaxPooling3D(pool_size=pool_size,
                               strides=strides)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(units=number_of_dense_units,
                    activation ='relu')(outputs)
    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)
    outputs = Dense(units=number_of_dense_units,
                    activation ='relu')(outputs)
    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)

    layer_activation = ''
    if mode == 'classification':
        layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('unrecognized mode.')

    outputs = Dense(units=number_of_classification_labels,
                    activation =layer_activation)(outputs)

    vgg_model = Model(inputs=inputs, outputs=outputs)

    return(vgg_model)

