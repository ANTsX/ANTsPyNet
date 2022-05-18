from tensorflow.keras.models import Sequential
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

    model = Sequential()
    model.add(Input(shape=(input_vector_size,)))

    number_of_filters = number_of_filters_at_base_layer
    for i in range(number_of_layers):

        model.add(Dense(units=number_of_filters))
        model.add(LeakyReLU(alpha=0.2))
        number_of_filters = int(number_of_filters / 2)

    if mode == "classification":
        model.add(Dense(units=number_of_classification_labels,
                        activation='softmax'))
    elif mode == "regression":
        model.add(Dense(units=1,
                        activation='linear'))
    else:
        raise ValueError("Unrecognized activation.")


    return(model)