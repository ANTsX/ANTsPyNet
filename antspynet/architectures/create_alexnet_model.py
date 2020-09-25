
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Lambda, Concatenate, Flatten, Dense,
                                    Conv2D, Conv2DTranspose, MaxPooling2D,
                                    ZeroPadding2D,
                                    Conv3D, Conv3DTranspose, MaxPooling3D,
                                    ZeroPadding3D)

def create_alexnet_model_2d(input_image_size,
                            number_of_classification_labels=1000,
                            number_of_dense_units=4096,
                            dropout_rate=0.0,
                            mode='classification'):
    """
    2-D implementation of the AlexNet deep learning architecture.

    Creates a keras model of the AlexNet deep learning architecture for image
    recognition based on the paper

    A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks.

    available here:

            http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/duggalrahul/AlexNet-Experiments-Keras/
            https://github.com/lunardog/convnets-keras/

    Arguments
    ---------
    input_image_size : tuple of length 3
        Used for specifying the input tensor shape.  The
        shape (or dimension) of that tensor is the image dimensions followed by
        the number of channels (e.g., red, green, and blue).  The batch size
        (i.e., number of training images) is not specified a priori.

    number_of_classification_labels : integer
        Number of segmentation labels.

    number_of_dense_units : integer
       Number of dense units.

    dropout_rate : scalar
       Optional regularization parameter between [0, 1]. Default = 0.0.

    mode : string
       'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 2-D Keras model defining the network.

    Example
    -------
    >>> model = create_alexnet_model_2d((128, 128, 1))
    >>> model.summary()

    """

    def split_tensor_2d(axis=1, ratio_split=1, id_split=0):

        def f(X):
           Xdims = K.int_shape(X)
           div = Xdims[axis] // ratio_split
           axis_split = slice(id_split * div, (id_split + 1) * div)

           if axis == 0:
               output = X[axis_split,:, :, :]
           elif axis == 1:
               output = X[:,axis_split, :, :]
           elif axis == 2:
               output = X[:,:, axis_split, :]
           elif axis == 3:
               output = X[:,:, :, axis_split]
           else:
               raise ValueError("axis specified is out of range.")

           return(output)

        def g(input_shape):
            output_shape = list(input_shape)
            output_shape[axis] = output_shape[axis] // ratio_split
            return(tuple(output_shape))

        return(Lambda(f, output_shape=lambda input_shape:g(input_shape)))

    def cross_channel_normalization_2d(alpha=1e-4, k=2, beta=0.75, n=5):

        def normalize_tensor_2d(X):

            X2 = K.square(X)

            half = n // 2

            extra_channels = K.spatial_2d_padding(
                K.permute_dimensions(X2, (1, 2, 3, 0)),
                padding=((0, 0), (half, half)))
            extra_channels = K.permute_dimensions(
                extra_channels, (3, 0, 1, 2))

            Xdims = K.int_shape(X)
            number_of_channels = int(Xdims[-1])

            scale = k
            for i in range(n):
                scale += alpha * extra_channels[:,:,:,i:(i + number_of_channels)]
            scale = scale ** beta

            return(X / scale)

        return(Lambda(normalize_tensor_2d, output_shape = lambda input_shape:input_shape))

    inputs = Input(shape = input_image_size)

    # Conv1
    outputs = Conv2D(filters = 96,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     activation='relu')(inputs)

    # Conv2
    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2))(outputs)
    normalization_layer = cross_channel_normalization_2d()
    outputs = normalization_layer(outputs)

    outputs = ZeroPadding2D(padding=(2, 2))(outputs)

    convolution_layer = Conv2D(filters=128,
                               kernel_size=(5, 5),
                               padding='same')(outputs)
    lambda_layers_conv2 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_2d(axis=3, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv2.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv2)

    # Conv3
    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2))(outputs)
    normalization_layer = cross_channel_normalization_2d()
    outputs = normalization_layer(outputs)

    outputs = ZeroPadding2D(padding=(2, 2))(outputs)
    outputs = Conv2D(filters=384,
                     kernel_size=(3, 3),
                     padding='same')(outputs)

    # Conv4
    outputs = ZeroPadding2D(padding=(2, 2))(outputs)
    convolution_layer = Conv2D(filters=192,
                               kernel_size=(3, 3),
                               padding='same')(outputs)
    lambda_layers_conv4 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_2d(axis=3, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv4.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv4)

    # Conv5
    outputs = ZeroPadding2D(padding=(2, 2))(outputs)
    normalization_layer = cross_channel_normalization_2d()
    outputs = normalization_layer(outputs)

    convolution_layer = Conv2D(filters=128,
                               kernel_size=(3, 3),
                               padding='same')(outputs)
    lambda_layers_conv5 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_2d(axis=3, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv5.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv5)

    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2))(outputs)
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
                    activation=layer_activation)(outputs)

    alexnet_model = Model(inputs=inputs, outputs=outputs)

    return(alexnet_model)


def create_alexnet_model_3d(input_image_size,
                            number_of_classification_labels=1000,
                            number_of_dense_units=4096,
                            dropout_rate=0.0,
                            mode='classification'):
    """
    3-D implementation of the AlexNet deep learning architecture.

    Creates a keras model of the AlexNet deep learning architecture for image
    recognition based on the paper

    A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks.

    available here:

            http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    This particular implementation was influenced by the following python
    implementation:

            https://github.com/duggalrahul/AlexNet-Experiments-Keras/
            https://github.com/lunardog/convnets-keras/

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The
        shape (or dimension) of that tensor is the image dimensions followed by
        the number of channels (e.g., red, green, and blue).  The batch size
        (i.e., number of training images) is not specified a priori.

    number_of_classification_labels : integer
        Number of segmentation labels.

    number_of_dense_units : integer
       Number of dense units.

    dropout_rate : scalar
       Optional regularization parameter between [0, 1]. Default = 0.0.

    mode : string
       'classification' or 'regression'.  Default = 'classification'.

    Returns
    -------
    Keras model
        A 3-D Keras model defining the network.

    Example
    -------
    >>> model = create_alexnet_model_3d((128, 128, 128, 1))
    >>> model.summary()

    """

    def split_tensor_3d(axis=1, ratio_split=1, id_split=0):

        def f(X):
           Xdims = K.int_shape(X)
           div = Xdims[axis] // ratio_split
           axis_split = slice(id_split * div, (id_split + 1) * div)

           if axis == 0:
               output = X[axis_split,:, :, :, :]
           elif axis == 1:
               output = X[:,axis_split, :, :, :]
           elif axis == 2:
               output = X[:,:, axis_split, :, :]
           elif axis == 3:
               output = X[:,:, :, axis_split, :]
           elif axis == 4:
               output = X[:,:, :, :, axis_split]
           else:
               raise ValueError("axis specified is out of range.")

           return(output)

        def g(input_shape):
            output_shape = list(input_shape)
            output_shape[axis] = output_shape[axis] // ratio_split
            return(tuple(output_shape))

        return(Lambda(f, output_shape=lambda input_shape:g(input_shape)))

    def cross_channel_normalization_3d(alpha=1e-4, k=2, beta=0.75, n=5):

        def normalize_tensor_3d(X):

            X2 = K.square(X)

            half = n // 2

            extra_channels = K.spatial_3d_padding(
                K.permute_dimensions(X2, (1, 2, 3, 4, 0)),
                padding=((0, 0), (0, 0), (half, half)))
            extra_channels = K.permute_dimensions(
                extra_channels, (4, 0, 1, 2, 3))

            Xdims = K.int_shape(X)
            number_of_channels = int(Xdims[-1])

            scale = k
            for i in range(n):
                scale += alpha * extra_channels[:,:,:,:,i:(i + number_of_channels)]
            scale = scale ** beta

            return(X / scale)

        return(Lambda(normalize_tensor_3d, output_shape = lambda input_shape:input_shape))

    inputs = Input(shape = input_image_size)

    # Conv1
    outputs = Conv3D(filters = 96,
                     kernel_size=(11, 11, 11),
                     strides=(4, 4, 4),
                     activation='relu')(inputs)

    # Conv2
    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2))(outputs)
    normalization_layer = cross_channel_normalization_3d()
    outputs = normalization_layer(outputs)

    outputs = ZeroPadding3D(padding=(2, 2, 2))(outputs)

    convolution_layer = Conv3D(filters=128,
                               kernel_size=(5, 5, 5),
                               padding='same')(outputs)
    lambda_layers_conv2 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_3d(axis=4, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv2.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv2)

    # Conv3
    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2))(outputs)
    normalization_layer = cross_channel_normalization_3d()
    outputs = normalization_layer(outputs)

    outputs = ZeroPadding3D(padding=(2, 2, 2))(outputs)
    outputs = Conv3D(filters=384,
                     kernel_size=(3, 3, 3),
                     padding='same')(outputs)

    # Conv4
    outputs = ZeroPadding3D(padding=(2, 2, 2))(outputs)
    convolution_layer = Conv3D(filters=192,
                               kernel_size=(3, 3, 3),
                               padding='same')(outputs)
    lambda_layers_conv4 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_3d(axis=4, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv4.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv4)

    # Conv5
    outputs = ZeroPadding3D(padding=(2, 2, 2))(outputs)
    normalization_layer = cross_channel_normalization_3d()
    outputs = normalization_layer(outputs)

    convolution_layer = Conv3D(filters=128,
                               kernel_size=(3, 3, 3),
                               padding='same')(outputs)
    lambda_layers_conv5 = [convolution_layer]
    for i in range(2):
        split_layer = split_tensor_3d(axis=4, ratio_split=2, id_split=i)(outputs)
        lambda_layers_conv5.append(split_layer)
    outputs = Concatenate()(lambda_layers_conv5)

    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2))(outputs)
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
                    activation=layer_activation)(outputs)

    alexnet_model = Model(inputs=inputs, outputs=outputs)

    return(alexnet_model)

