
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def create_autoencoder_model(number_of_units_per_layer,
                             activation='relu',
                             initializer='glorot_uniform'
                            ):
    """
    2-D implementation of the Vgg deep learning architecture.

    Builds an autoencoder based on the specified array definining the
    number of units in the encoding branch.  Ported to Keras R from the
    Keras python implementation here:

    https://github.com/XifengGuo/DEC-keras

    Arguments
    ---------
    number_of_units_per_layer : tuple
        A tuple defining the number of units in the encoding branch.

    activation : string
        Activation type for the dense layers

    initializer : string
        Initializer type for the dense layers

    Returns
    -------
    Keras model
        An encoder and autoencoder Keras model.

    Example
    -------
    >>> model = create_autoencoder_model((784, 500, 500, 2000, 10))
    >>> model.summary()
    """

    number_of_encoding_layers = len(number_of_units_per_layer) - 1

    inputs = Input(shape=(number_of_units_per_layer[0],))

    encoder = inputs

    for i in range(number_of_encoding_layers - 1):
        encoder = Dense(units=number_of_units_per_layer[i + 1],
                        activation=activation,
                        kernel_initializer=initializer)(encoder)

    encoder = Dense(units=number_of_units_per_layer[-1])(encoder)

    autoencoder = encoder

    for i in range(number_of_encoding_layers-1, 0, -1):
        autoencoder = Dense(units=number_of_units_per_layer[i],
                            activation=activation,
                            kernel_initializer=initializer)(autoencoder)

    autoencoder = Dense(units=number_of_units_per_layer[0],
                        kernel_initializer=initializer)(autoencoder)

    encoder_model = Model(inputs=inputs, outputs=encoder)
    autoencoder_model = Model(inputs=inputs, outputs=autoencoder)

    return(autoencoder_model, encoder_model)
