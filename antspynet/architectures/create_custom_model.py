from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Add, Activation, AveragePooling3D, BatchNormalization,
                          Conv3D, Dropout, Input, MaxPooling3D, ReLU, ZeroPadding3D)
from antspynet.utilities import LogSoftmax                          

def create_simple_fully_convolutional_network_model_3d(input_image_size,
                                                       number_of_filters_per_layer=(32, 64, 128, 256, 256, 64),
                                                       number_of_bins=40,
                                                       dropout_rate=0.5):
    """
    Implementation of the "SCFN" architecture for Brain/Gender prediction

    Creates a keras model implementation of the Simple Fully Convolutional
    Network model from the FMRIB group:

       https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).
    number_of_filters_per_layer : array 
        number of filters for the convolutional layers.
    number_of_bins : integer
        number of bins for final softmax output.
    dropout_rate : float between 0 and 1
        Optional dropout rate before final convolution layer. 

    Returns
    -------
    Keras model
        A 3-D keras model.

    Example
    -------
    >>> model = create_simple_fully_convolutional_network_model_3d((None, None, None, 1))
    >>> model.summary()
    """

    number_of_layers = len(number_of_filters_per_layer)

    inputs = Input(shape=input_image_size)
 
    outputs = inputs
    for i in range(number_of_layers):
        if i < number_of_layers - 1:
            outputs = Conv3D(filters=number_of_filters_per_layer[i],
                             kernel_size=(3, 3, 3),
                             padding='valid')(outputs)
            outputs = ZeroPadding3D(padding=(1, 1, 1))(outputs)                 
            outputs = BatchNormalization(momentum=0.1,
                                         epsilon=1e-5)(outputs)
            outputs = MaxPooling3D(pool_size=(2, 2, 2),
                                   strides=(2, 2, 2))(outputs)
        else:
            outputs = Conv3D(filters=number_of_filters_per_layer[i],
                             kernel_size=(1, 1, 1),
                             padding='valid')(outputs)
            outputs = BatchNormalization(momentum=0.1,
                                         epsilon=1e-5)(outputs)
        outputs = ReLU()(outputs)

    outputs = AveragePooling3D(pool_size=(5, 6, 5),
                               strides=(5, 6, 5))(outputs)

    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)

    outputs = Conv3D(filters=number_of_bins,
                     kernel_size=(1, 1, 1),
                     padding='valid')(outputs)
    outputs = LogSoftmax()(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model

