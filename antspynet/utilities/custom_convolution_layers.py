import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import conv_utils
from keras.layers import InputSpec, Conv2D, Conv3D

class PartialConv2D(Conv2D):

    """
    Implementation of the 2-D partial convolution layer based on

    https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/pconv_layer.py

    Returns
    -------
    Keras custom layer

    """

    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        super(PartialConv2D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        self.eps = eps

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = (*self.kernel_size, self.input_dim, self.filters)

        self.kernel = self.add_weight(name="kernel",
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)

        self.mask_kernel = tf.Variable(initial_value=tf.ones(kernel_shape),
                                       trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):

        features = inputs[0]
        feature_mask = inputs[1]
        if feature_mask.shape[-1] == 1:
            feature_mask = tf.repeat(feature_mask, tf.shape(features)[-1], axis=-1)

        features = tf.multiply(features, feature_mask)
        features = K.conv2d(features,
            self.kernel,
            strides=self.strides,
            padding="same",
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        norm = K.conv2d(feature_mask,
           self.mask_kernel,
           strides=self.strides,
           padding="same",
           data_format=self.data_format,
           dilation_rate=self.dilation_rate
        )

        # The following commented normalization code wasn't producing expected results
        # for inpainting.  So I compared the PartialConv2D with an all-ones mask vs. a
        # conventional Conv2D in a simple U-net model trained to predict an output from
        # an identical input using mse as the loss (e.g., input r16 slice and get out the
        # same r16 slice).  As expected, the Conv2D option resulted in expected results.
        # In contrast, the PartialConv2D output looked like a blurred version of the
        # input.  I traced it to the following two normalization lines.  From what I'm
        # guessing, the division operation incorporating the input mask causes the
        # back-propagation gradient to die.  However, given that the normalization kernel
        # will have a finite set of values in the range {0,...,mask_fanin}, I replaced the
        # earlier convolution divide with a search-and-replace.  The replacement code
        # seems to not have the same issue.
        #
        # norm = tf.math.divide(norm, mask_fanin)
        # features = tf.math.divide_no_nan(features, norm)

        feature_mask_fanin = self.kernel_size[0] * self.kernel_size[1]
        for i in range(2, feature_mask_fanin+1):
            features = tf.where(tf.equal(norm, tf.constant(i, dtype=tf.float32)),
                                    tf.math.divide(features, tf.constant(i, dtype=tf.float32)),
                                    features)

        if self.use_bias:
            features = tf.add(features, self.bias)

        # Apply activations on the image
        if self.activation is not None:
            features = self.activation(features)

        feature_mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)

        return [features, feature_mask]

    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        if self.data_format == 'channels_last':
            space = feature_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (feature_shape[0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (feature_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]



class PartialConv3D(Conv3D):

    """
    Implementation of the 3-D partial convolution layer based on

    https://github.com/MathiasGruber/PConv-Keras/blob/master/libs/pconv_layer.py

    Returns
    -------
    Keras custom layer

    """

    def __init__(self,
                 eps=1e-6,
                 **kwargs):
        super(PartialConv3D, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=5), InputSpec(ndim=5)]
        self.eps = eps

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = (*self.kernel_size, self.input_dim, self.filters)

        self.kernel = self.add_weight(name="kernel",
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)

        self.mask_kernel = tf.Variable(initial_value=tf.ones(kernel_shape),
                                       trainable=False)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, mask=None):

        features = inputs[0]
        feature_mask = inputs[1]
        if feature_mask.shape[-1] == 1:
            feature_mask = tf.repeat(feature_mask, tf.shape(features)[-1], axis=-1)

        features = tf.multiply(features, feature_mask)
        features = K.conv3d(features,
            self.kernel,
            strides=self.strides,
            padding="same",
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        norm = K.conv3d(feature_mask,
           self.mask_kernel,
           strides=self.strides,
           padding="same",
           data_format=self.data_format,
           dilation_rate=self.dilation_rate
        )

        feature_mask_fanin = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        for i in range(2, feature_mask_fanin+1):
            features = tf.where(tf.equal(norm, tf.constant(i, dtype=tf.float32)),
                                    tf.math.divide(features, tf.constant(i, dtype=tf.float32)),
                                    features)

        if self.use_bias:
            features = tf.add(features, self.bias)

        # Apply activations on the image
        if self.activation is not None:
            features = self.activation(features)

        feature_mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)

        return [features, feature_mask]

    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape

        if self.data_format == 'channels_last':
            space = feature_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (feature_shape[0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (feature_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]



