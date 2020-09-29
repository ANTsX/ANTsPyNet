
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Concatenate
from tensorflow.keras import initializers


class AttentionLayer2D(Layer):

    """
    Attention layer (2-D) from the self attention GAN

    taken from the following python implementation

    https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras

    based on the following paper:

    https://arxiv.org/abs/1805.08318

    Arguments
    ---------
    number_of_channels : integer
        Number of channels

    Returns
    -------
    Layer
        A keras layer

    """

    def __init__(self, number_of_channels, **kwargs):

        super(AttentionLayer2D, self).__init__(**kwargs)

        self.number_of_channels = number_of_channels
        self.number_of_filters_f_g = self.number_of_channels // 8
        self.number_of_filters_h = self.number_of_channels

    def build(self, input_shape):

        kernel_shape_f_g = (1, 1) + (self.number_of_channels, self.number_of_filters_f_g)
        kernel_shape_h = (1, 1) + (self.number_of_channels, self.number_of_filters_h)

        self.gamma = self.add_weight(shape=[1],
                                     initializer=initializers.zeros(),
                                     trainable=True,
                                     name="gamma")
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_f")
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_g")
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_h")
        self.bias_f = self.add_weight(shape=(self.number_of_filters_f_g,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_f")
        self.bias_g = self.add_weight(shape=(self.number_of_filters_f_g,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_g")
        self.bias_h = self.add_weight(shape=(self.number_of_filters_h,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_h")

        super(AttentionLayer2D, self).build(input_shape)

        self.input_spec = InputSpec(ndim=4, axes={3: input_shape[-1]})
        self.built = True

    def call(self, inputs, mask=None):

        def flatten(x):
            input_shape = K.shape(x)
            output_shape = (input_shape[0], input_shape[1] * input_shape[2], input_shape[3])
            x_flat = K.reshape(x, shape=output_shape)
            return( x_flat )

        f = K.conv2d(inputs, kernel=self.kernel_f, strides=(1, 1), padding='same')
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(inputs, kernel=self.kernel_g, strides=(1, 1), padding='same')
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(inputs, kernel=self.kernel_h, strides=(1, 1), padding='same')
        h = K.bias_add(h, self.bias_h)

        f_flat = flatten(f)
        g_flat = flatten(g)
        h_flat = flatten(h)

        s = tf.matmul(g_flat, f_flat, transpose_b = True)
        beta = K.softmax(s, axis=-1)
        o = K.reshape(K.batch_dot(beta, h_flat), shape=K.shape(inputs))

        x = self.gamma * o + inputs
        return(x)

    def compute_output_shape(self, input_shape):
        return(input_shape)

    def get_config(self):
        config = {"number_of_channels": self.number_of_channels}
        base_config = super(AttentionLayer2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionLayer3D(Layer):

    """
    Attention layer (3-D) from the self attention GAN

    taken from the following python implementation

    https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras

    based on the following paper:

    https://arxiv.org/abs/1805.08318

    Arguments
    ---------
    number_of_channels : integer
        Number of channels

    Returns
    -------
    Layer
        A keras layer

    Example
    -------
    >>> input_shape = (100, 100, 3)
    >>> input = Input(shape=input_shape)
    >>> number_of_filters = 64
    >>> outputs = Conv2D(filters=number_of_filters, kernel_size=2)(input)
    >>> outputs = AttentionLayer2D(number_of_channels=number_of_filters)(outputs)
    >>> model = Model(inputs=input, outputs=outputs)

    """

    def __init__(self, number_of_channels, **kwargs):

        super(AttentionLayer3D, self).__init__(**kwargs)

        self.number_of_channels = number_of_channels
        self.number_of_filters_f_g = self.number_of_channels // 8
        self.number_of_filters_h = self.number_of_channels

    def build(self, input_shape):

        kernel_shape_f_g = (1, 1, 1) + (self.number_of_channels, self.number_of_filters_f_g)
        kernel_shape_h = (1, 1, 1) + (self.number_of_channels, self.number_of_filters_h)

        self.gamma = self.add_weight(shape=[1],
                                     initializer=initializers.zeros(),
                                     trainable=True,
                                     name="gamma")
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_f")
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_g")
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer=initializers.glorot_uniform(),
                                        trainable=True,
                                        name="kernel_h")
        self.bias_f = self.add_weight(shape=(self.number_of_filters_f_g,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_f")
        self.bias_g = self.add_weight(shape=(self.number_of_filters_f_g,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_g")
        self.bias_h = self.add_weight(shape=(self.number_of_filters_h,),
                                      initializer=initializers.zeros(),
                                      trainable=True,
                                      name="bias_h")

        super(AttentionLayer3D, self).build(input_shape)

        self.input_spec = InputSpec(ndim=5, axes={4: input_shape[-1]})
        self.built = True

    def call(self, inputs, mask=None):

        def flatten(x):
            input_shape = K.shape(x)
            output_shape = (input_shape[0], input_shape[1] * input_shape[2] * input_shape[3], input_shape[4])
            x_flat = K.reshape(x, shape=output_shape)
            return( x_flat )

        f = K.conv3d(inputs, kernel=self.kernel_f, strides=(1, 1, 1), padding='same')
        f = K.bias_add(f, self.bias_f)
        g = K.conv3d(inputs, kernel=self.kernel_g, strides=(1, 1, 1), padding='same')
        g = K.bias_add(g, self.bias_g)
        h = K.conv3d(inputs, kernel=self.kernel_h, strides=(1, 1, 1), padding='same')
        h = K.bias_add(h, self.bias_h)

        f_flat = flatten(f)
        g_flat = flatten(g)
        h_flat = flatten(h)

        s = tf.matmul(g_flat, f_flat, transpose_b = True)
        beta = K.softmax(s, axis=-1)
        o = K.reshape(K.batch_dot(beta, h_flat), shape=K.shape(inputs))

        x = self.gamma * o + inputs
        return(x)

    def compute_output_shape(self, input_shape):
        return(input_shape)

    def get_config(self):
        config = {"number_of_channels": self.number_of_channels}
        base_config = super(AttentionLayer3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

