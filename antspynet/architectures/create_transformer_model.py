import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Add, Dense, Dropout, Flatten,
                                     Input, LayerNormalization, MultiHeadAttention)
from antspynet.utilities import (ExtractPatches2D, ExtractPatches3D, EncodePatches,
                                 ExtractConvolutionalPatches2D, ExtractConvolutionalPatches3D,
                                 StochasticDepth)
import numpy as np

def multilayer_perceptron(x, hidden_units, dropout_rate=0.0):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate)(x)
    return x

def create_vision_transformer_model_2d(input_image_size,
                                       number_of_classification_labels=1000,
                                       patch_size=6,
                                       number_of_transformer_layers=8,
                                       transformer_units=[128, 64],
                                       projection_dimension=64,
                                       number_of_attention_heads=4,
                                       mlp_head_units=[2048, 1024],
                                       dropout_rate=0.5):
    """
    Implementation of the Vision transformer architecture.

       https://keras.io/examples/vision/image_classification_with_vision_transformer/


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : int
        Number of classification labels.

    patch_size : int
        Size of a single patch dimension.

    number_of_transformer_layers : int
        Number of transformer layers.

    transformer_units : tuple or list
        Size of the hidden units in the layers of the MLP.

    projection_dimension : int
        Multi-head attention layer parameter

    mlp_head_units : tuple or list
        Size of the dense layers of the final classifier.

    dropout_rate : float between 0 and 1
        Dropout rate of the multilayer perceptron and the previous dropout layer.

    Returns
    -------
    Keras model
        A 2-D keras model.

    Example
    -------
    >>> model = create_vision_transformer_model_2d((224, 224, 1))
    >>> model.summary()
    """

    inputs = Input(shape=input_image_size)

    patches = ExtractPatches2D(patch_size)(inputs)
    number_of_patches = ((input_image_size[1] * input_image_size[2]) // (patch_size ** 2))
    encoded_patches = EncodePatches(number_of_patches,
                                    projection_dimension)(patches)

    for _ in range(number_of_transformer_layers):

        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = MultiHeadAttention(num_heads=number_of_attention_heads,
                                              key_dim=projection_dimension,
                                              dropout=dropout_rate/5.0)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = multilayer_perceptron(x3,
                                   hidden_units=transformer_units,
                                   dropout_rate=dropout_rate/5.0)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(dropout_rate)(representation)

    features = multilayer_perceptron(representation,
                                     hidden_units=mlp_head_units,
                                     dropout_rate=dropout_rate)

    outputs = Dense(number_of_classification_labels)(features)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_vision_transformer_model_3d(input_image_size,
                                       number_of_classification_labels=1000,
                                       patch_size=6,
                                       number_of_transformer_layers=8,
                                       transformer_units=[128, 64],
                                       projection_dimension=64,
                                       number_of_attention_heads=4,
                                       mlp_head_units=[2048, 1024],
                                       dropout_rate=0.5):
    """
    Implementation of the Vision transformer architecture.

       https://keras.io/examples/vision/image_classification_with_vision_transformer/


    Arguments
    ---------
    input_image_size : tuple of length 5
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : int
        Number of classification labels.

    patch_size : int
        Size of a single patch dimension.

    number_of_transformer_layers : int
        Number of transformer layers.

    transformer_units : tuple or list
        Size of the hidden units in the layers of the MLP.

    projection_dimension : int
        Multi-head attention layer parameter

    mlp_head_units : tuple or list
        Size of the dense layers of the final classifier.

    dropout_rate : float between 0 and 1
        Dropout rate of the multilayer perceptron and the previous dropout layer.

    Returns
    -------
    Keras model
        A 3-D keras model.

    Example
    -------
    >>> model = create_vision_transformer_model_3d(((224, 224, 224, 1))
    >>> model.summary()
    """

    inputs = Input(shape=input_image_size)

    patches = ExtractPatches3D(patch_size)(inputs)
    number_of_patches = ((input_image_size[1] * input_image_size[2]) // (patch_size ** 2))
    encoded_patches = EncodePatches(number_of_patches,
                                    projection_dimension)(patches)

    for _ in range(number_of_transformer_layers):

        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = MultiHeadAttention(num_heads=number_of_attention_heads,
                                              key_dim=projection_dimension,
                                              dropout=dropout_rate/5.0)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = multilayer_perceptron(x3,
                                   hidden_units=transformer_units,
                                   dropout_rate=dropout_rate/5.0)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(dropout_rate)(representation)

    features = multilayer_perceptron(representation,
                                     hidden_units=mlp_head_units,
                                     dropout_rate=dropout_rate)

    outputs = Dense(number_of_classification_labels)(features)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_compact_convolutional_transformer_model_2d(input_image_size,
                                                      number_of_classification_labels=1000,
                                                      number_of_transformer_layers=8,
                                                      transformer_units=[128, 128],
                                                      projection_dimension=64,
                                                      number_of_attention_heads=4,
                                                      stochastic_depth_rate=0.1):
    """
    Implementation of the Vision transformer architecture.

       https://keras.io/examples/vision/cct/


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : int
        Number of classification labels.

    patch_size : int
        Size of a single patch dimension.

    number_of_transformer_layers : int
        Number of transformer layers.

    transformer_units : tuple or list
        Size of the hidden units in the layers of the MLP.

    projection_dimension : int
        Multi-head attention layer parameter

    stochastic_depth_rate : float between 0 and 1
        Dropout rate of the stochastic depth layer

    Returns
    -------
    Keras model
        A 2-D keras model.

    Example
    -------
    >>> model = antspynet.create_compact_convolutional_transformer_model_2d((224, 224, 1))
    >>> model.summary()
    """

    inputs = Input(shape=input_image_size)

    ExtractPatches = ExtractConvolutionalPatches2D(kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   pooling_kernel_size=3,
                                                   pooling_stride=2,
                                                   number_of_filters=[64, 128],
                                                   do_positional_embedding=True)
    encoded_patches = ExtractPatches(inputs)

    # Apply positional embedding.
    positional_embedding, sequence_length = ExtractPatches.positional_embedding(input_image_size)
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = positional_embedding(positions)
    encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, number_of_transformer_layers)]

    for i in range(number_of_transformer_layers):

        x1 = LayerNormalization(epsilon=1e-5)(encoded_patches)

        attention_output = MultiHeadAttention(num_heads=number_of_attention_heads,
                                              key_dim=projection_dimension,
                                              dropout=0.1)(x1, x1)
        attention_output = StochasticDepth(dpr[i])(attention_output)

        x2 = Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=1e-5)(x2)
        x3 = multilayer_perceptron(x3,
                                   hidden_units=transformer_units,
                                   dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(attention_weights, representation, transpose_a=True)
    weighted_representation = tf.squeeze(weighted_representation, -2)

    outputs = Dense(number_of_classification_labels)(weighted_representation)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def create_compact_convolutional_transformer_model_3d(input_image_size,
                                                      number_of_classification_labels=1000,
                                                      number_of_transformer_layers=8,
                                                      transformer_units=[128, 128],
                                                      projection_dimension=64,
                                                      number_of_attention_heads=4,
                                                      stochastic_depth_rate=0.1):
    """
    Implementation of the Vision transformer architecture.

       https://keras.io/examples/vision/cct/


    Arguments
    ---------
    input_image_size : tuple of length 5
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_classification_labels : int
        Number of classification labels.

    patch_size : int
        Size of a single patch dimension.

    number_of_transformer_layers : int
        Number of transformer layers.

    transformer_units : tuple or list
        Size of the hidden units in the layers of the MLP.

    projection_dimension : int
        Multi-head attention layer parameter

    stochastic_depth_rate : float between 0 and 1
        Dropout rate of the stochastic depth layer

    Returns
    -------
    Keras model
        A 3-D keras model.

    Example
    -------
    >>> model = antspynet.create_compact_convolutional_transformer_model_3d((224, 224, 224, 1))
    >>> model.summary()
    """

    inputs = Input(shape=input_image_size)

    ExtractPatches = ExtractConvolutionalPatches3D(kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   pooling_kernel_size=3,
                                                   pooling_stride=2,
                                                   number_of_filters=[64, 128],
                                                   do_positional_embedding=True)
    encoded_patches = ExtractPatches(inputs)

    # Apply positional embedding.
    positional_embedding, sequence_length = ExtractPatches.positional_embedding(input_image_size)
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = positional_embedding(positions)
    encoded_patches += position_embeddings

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, number_of_transformer_layers)]

    for i in range(number_of_transformer_layers):

        x1 = LayerNormalization(epsilon=1e-5)(encoded_patches)

        attention_output = MultiHeadAttention(num_heads=number_of_attention_heads,
                                              key_dim=projection_dimension,
                                              dropout=0.1)(x1, x1)
        attention_output = StochasticDepth(dpr[i])(attention_output)

        x2 = Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=1e-5)(x2)
        x3 = multilayer_perceptron(x3,
                                   hidden_units=transformer_units,
                                   dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(attention_weights, representation, transpose_a=True)
    weighted_representation = tf.squeeze(weighted_representation, -2)

    outputs = Dense(number_of_classification_labels)(weighted_representation)

    model = Model(inputs=inputs, outputs=outputs)

    return model

