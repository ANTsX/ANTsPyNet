from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU

def create_dense_model(input_vector_size,
                       number_of_filters_at_base_layer=512,
                       number_of_layers=2,
                       mode='classification',
                       number_of_classification_labels=1000
                      ):
    """

    Simple multilayer dense network.

    Arguments
    ---------
    input_vector size : integer
        Specifies the length of the input vector.

    number_of_filters_at_base_layer : integer
        number of filters at the initial dense layer.  This number is halved for
        each subsequent layer.

    number_of_layers : integer
        Number of dense layers defining the model.

    mode : string
        "regression" or "classification".

    number_of_classification_labels : integer
        Specifies output for "classification" networks.

    Returns
    -------
    Keras model
        A Keras model defining the network.

    """

    input = Input(shape=(input_vector_size,))

    output = input

    number_of_filters = number_of_filters_at_base_layer
    for i in range(number_of_layers):

        output = Dense(units=number_of_filters)(output)
        output = LeakyReLU(alpha=0.2)(output)
        number_of_filters = int(number_of_filters / 2)

    if mode == "classification":
        output = Dense(units=number_of_classification_labels, activation='softmax')(output)
    elif mode == "regression":
        output = Dense(units=1, activation='linear')(output)
    else:
        raise ValueError("Unrecognized activation.")

    model = Model(inputs=input, outputs=output)

    return(model)