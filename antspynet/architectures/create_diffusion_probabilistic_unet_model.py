import tensorflow as tf

import math

from tensorflow import keras

from keras.layers import (Add, Concatenate, Conv2D, Conv3D, Dense, GroupNormalization,
                          Input, Layer, UpSampling2D, UpSampling3D)
from keras.initializers import VarianceScaling
from keras.activations import swish
from keras import Model


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return VarianceScaling(scale, mode="fan_avg", distribution="uniform")

class TimeEmbedding(Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

def TimeMLP(units, activation_function=swish):
    def apply(inputs):
        temb = Dense(units,
                     activation=activation_function,
                     kernel_initializer=kernel_init(1.0))(inputs)
        temb = Dense(units,
                     kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

def create_diffusion_probabilistic_unet_model_2d(input_image_size,
                                                 number_of_filters=(64, 128, 256, 512),
                                                 number_of_outputs=1,
                                                 has_attention=[False, False, True, True],
                                                 number_of_residual_blocks=2,
                                                 number_of_normalization_groups=8,
                                                 interpolation="nearest",
                                                 activation_function=swish):

    """
    2-D implementation of the U-net architecture for denoising diffusion probabilistic
    modeling taken from:

        https://github.com/keras-team/keras-io/blob/master/examples/generative/ddpm.py

    Arguments
    ---------
    input_image_size : tuple of length 3
        Tuple of ints of length 3 specifying 2-D image size and channel size.

    number_of_filters: tuple
        Specifies the filter schedule.  Defaults to the number of filters used in
        the paper.

    Returns
    -------
    Keras model
        A 2-D keras model defining the U-net network.

    Example
    -------
    >>>
    """

    class AttentionBlock2D(Layer):
        def __init__(self, units, groups=8, **kwargs):
            self.units = units
            self.groups = groups
            super().__init__(**kwargs)

            self.norm = GroupNormalization(groups=groups)
            self.query = Dense(units,
                               kernel_initializer=kernel_init(1.0))
            self.key = Dense(units,
                             kernel_initializer=kernel_init(1.0))
            self.value = Dense(units,
                               kernel_initializer=kernel_init(1.0))
            self.proj = Dense(units,
                              kernel_initializer=kernel_init(0.0))

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            scale = tf.cast(self.units, tf.float32) ** (-0.5)

            inputs = self.norm(inputs)
            q = self.query(inputs)
            k = self.key(inputs)
            v = self.value(inputs)

            attention_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
            attention_score = tf.reshape(attention_score, [batch_size, height, width, height * width])

            attention_score = tf.nn.softmax(attention_score, -1)
            attention_score = tf.reshape(attention_score, [batch_size, height, width, height, width])

            proj = tf.einsum("bhwHW,bHWc->bhwc", attention_score, v)
            proj = self.proj(proj)
            return inputs + proj

    def ResidualBlock2D(width, groups=8, activation_function=swish):
        def apply(inputs):
            x, t = inputs
            input_width = x.shape[3]

            if input_width == width:
                residual = x
            else:
                residual = Conv2D(width,
                                  kernel_size=1,
                                  kernel_initializer=kernel_init(1.0))(x)

            temb = activation_function(t)
            temb = Dense(width,
                         kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

            x = GroupNormalization(groups=groups)(x)
            x = activation_function(x)
            x = Conv2D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(1.0)
            )(x)

            x = Add()([x, temb])
            x = GroupNormalization(groups=groups)(x)
            x = activation_function(x)

            x = Conv2D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(0.0))(x)
            x = Add()([x, residual])
            return x
        return apply


    def DownSample2D(width):
        def apply(x):
            x = Conv2D(width,
                       kernel_size=3,
                       strides=2,
                       padding="same",
                       kernel_initializer=kernel_init(1.0))(x)
            return x
        return apply


    def UpSample2D(width, interpolation="nearest"):
        def apply(x):
            x = UpSampling2D(size=2,
                             interpolation=interpolation)(x)
            x = Conv2D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(1.0))(x)
            return x
        return apply

    image_input = Input(shape=input_image_size)
    time_input_tensor = keras.Input(shape=(), dtype=tf.int64)

    x = Conv2D(number_of_filters[0],
               kernel_size=(3, 3),
               padding="same",
               kernel_initializer=kernel_init(1.0))(image_input)

    temb = TimeEmbedding(dim=number_of_filters[0] * 4)(time_input_tensor)
    temb = TimeMLP(units=number_of_filters[0] * 4,
                   activation_function=swish)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(number_of_filters)):
        for _ in range(number_of_residual_blocks):
            x = ResidualBlock2D(number_of_filters[i],
                                groups=number_of_normalization_groups,
                                activation_function=activation_function)([x, temb])
            if has_attention[i]:
                x = AttentionBlock2D(number_of_filters[i],
                                     groups=number_of_normalization_groups)(x)
            skips.append(x)

        if number_of_filters[i] != number_of_filters[-1]:
            x = DownSample2D(number_of_filters[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock2D(number_of_filters[-1],
                        groups=number_of_normalization_groups,
                        activation_function=activation_function)([x, temb])
    x = AttentionBlock2D(number_of_filters[-1],
                         groups=number_of_normalization_groups)(x)
    x = ResidualBlock2D(number_of_filters[-1],
                        groups=number_of_normalization_groups,
                        activation_function=activation_function)([x, temb])

    # UpBlock
    for i in reversed(range(len(number_of_filters))):
        for _ in range(number_of_residual_blocks + 1):
            x = Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock2D(number_of_filters[i],
                                groups=number_of_normalization_groups,
                                activation_function=activation_function)([x, temb])
            if has_attention[i]:
                x = AttentionBlock2D(number_of_filters[i],
                                     groups=number_of_normalization_groups)(x)

        if i != 0:
            x = UpSample2D(number_of_filters[i],
                           interpolation=interpolation)(x)

    # End block
    x = GroupNormalization(groups=number_of_normalization_groups)(x)
    x = activation_function(x)
    x = Conv2D(number_of_outputs,
               kernel_size=(3, 3),
               padding="same",
               kernel_initializer=kernel_init(0.0))(x)
    return Model(inputs=[image_input, time_input_tensor], outputs=x)


def create_diffusion_probabilistic_unet_model_3d(input_image_size,
                                                 number_of_filters=(64, 128, 256, 512),
                                                 number_of_outputs=1,
                                                 has_attention=[False, False, True, True],
                                                 number_of_residual_blocks=2,
                                                 number_of_normalization_groups=8,
                                                 activation_function=swish):

    """
    3-D implementation of the U-net architecture for denoising diffusion probabilistic
    modeling taken from:

        https://github.com/keras-team/keras-io/blob/master/examples/generative/ddpm.py

    Arguments
    ---------
    input_image_size : tuple of length 3
        Tuple of ints of length 3 specifying 2-D image size and channel size.

    number_of_filters: tuple
        Specifies the filter schedule.  Defaults to the number of filters used in
        the paper.

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>>
    """

    class AttentionBlock3D(Layer):
        def __init__(self, units, groups=8, **kwargs):
            self.units = units
            self.groups = groups
            super().__init__(**kwargs)

            self.norm = GroupNormalization(groups=groups)
            self.query = Dense(units,
                               kernel_initializer=kernel_init(1.0))
            self.key = Dense(units,
                             kernel_initializer=kernel_init(1.0))
            self.value = Dense(units,
                               kernel_initializer=kernel_init(1.0))
            self.proj = Dense(units,
                              kernel_initializer=kernel_init(0.0))

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            depth = tf.shape(inputs)[3]
            scale = tf.cast(self.units, tf.float32) ** (-0.5)

            inputs = self.norm(inputs)
            q = self.query(inputs)
            k = self.key(inputs)
            v = self.value(inputs)

            attention_score = tf.einsum("bhwdc, bHWDc->bhwdHW", q, k) * scale
            attention_score = tf.reshape(attention_score, [batch_size, height, width, depth, height * width * depth])

            attention_score = tf.nn.softmax(attention_score, -1)
            attention_score = tf.reshape(attention_score, [batch_size, height, width, depth, height, width, depth])

            proj = tf.einsum("bhwdHWD,bHWDc->bhwdc", attention_score, v)
            proj = self.proj(proj)
            return inputs + proj

    def ResidualBlock3D(width, groups=8, activation_function=swish):
        def apply(inputs):
            x, t = inputs
            input_width = x.shape[4]

            if input_width == width:
                residual = x
            else:
                residual = Conv3D(width,
                                  kernel_size=1,
                                  kernel_initializer=kernel_init(1.0))(x)

            temb = activation_function(t)
            temb = Dense(width,
                         kernel_initializer=kernel_init(1.0))(temb)[:, None, None, None, :]

            x = GroupNormalization(groups=groups)(x)
            x = activation_function(x)
            x = Conv3D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(1.0)
            )(x)

            x = Add()([x, temb])
            x = GroupNormalization(groups=groups)(x)
            x = activation_function(x)

            x = Conv3D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(0.0))(x)
            x = Add()([x, residual])
            return x
        return apply


    def DownSample3D(width):
        def apply(x):
            x = Conv3D(width,
                       kernel_size=3,
                       strides=2,
                       padding="same",
                       kernel_initializer=kernel_init(1.0))(x)
            return x
        return apply


    def UpSample3D(width):
        def apply(x):
            x = UpSampling3D(size=2)(x)
            x = Conv3D(width,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer=kernel_init(1.0))(x)
            return x
        return apply

    image_input = Input(shape=input_image_size)
    time_input_tensor = keras.Input(shape=(), dtype=tf.int64)

    x = Conv3D(number_of_filters[0],
               kernel_size=(3, 3, 3),
               padding="same",
               kernel_initializer=kernel_init(1.0))(image_input)

    temb = TimeEmbedding(dim=number_of_filters[0] * 4)(time_input_tensor)
    temb = TimeMLP(units=number_of_filters[0] * 4,
                   activation_function=swish)(temb)

    skips = [x]

    # DownBlock
    for i in range(len(number_of_filters)):
        for _ in range(number_of_residual_blocks):
            x = ResidualBlock3D(number_of_filters[i],
                                groups=number_of_normalization_groups,
                                activation_function=activation_function)([x, temb])
            if has_attention[i]:
                x = AttentionBlock3D(number_of_filters[i],
                                     groups=number_of_normalization_groups)(x)
            skips.append(x)

        if number_of_filters[i] != number_of_filters[-1]:
            x = DownSample3D(number_of_filters[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock3D(number_of_filters[-1],
                        groups=number_of_normalization_groups,
                        activation_function=activation_function)([x, temb])
    x = AttentionBlock3D(number_of_filters[-1],
                         groups=number_of_normalization_groups)(x)
    x = ResidualBlock3D(number_of_filters[-1],
                        groups=number_of_normalization_groups,
                        activation_function=activation_function)([x, temb])

    # UpBlock
    for i in reversed(range(len(number_of_filters))):
        for _ in range(number_of_residual_blocks + 1):
            x = Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock3D(number_of_filters[i],
                                groups=number_of_normalization_groups,
                                activation_function=activation_function)([x, temb])
            if has_attention[i]:
                x = AttentionBlock3D(number_of_filters[i],
                                     groups=number_of_normalization_groups)(x)

        if i != 0:
            x = UpSample3D(number_of_filters[i])(x)

    # End block
    x = GroupNormalization(groups=number_of_normalization_groups)(x)
    x = activation_function(x)
    x = Conv2D(number_of_outputs,
               kernel_size=(3, 3),
               padding="same",
               kernel_initializer=kernel_init(0.0))(x)
    return Model(inputs=[image_input, time_input_tensor], outputs=x)
