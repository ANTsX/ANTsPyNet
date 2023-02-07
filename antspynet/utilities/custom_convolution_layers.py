import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.utils import conv_utils
from keras.layers import InputSpec, Conv2D, Conv3D

class PartialConv2D(Conv2D):

    def __init__(self,
                 *args,
                 eps=1e-6,
                 **kwargs):
        super(PartialConv2D, self).__init__(*args, **kwargs)
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
        mask_fanin = self.kernel_size[0] * self.kernel_size[1]
        self.mask_kernel = tf.ones(kernel_shape) / tf.cast(mask_fanin, 'float32')

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

        super(PartialConv2D, self).build(input_shape[0])

    def call(self, inputs, mask=None):

        features = inputs[0]
        mask = inputs[1]
        if mask.shape[-1] == 1:
            mask = tf.repeat(mask, tf.shape(features)[-1], axis=-1)

        features = tf.multiply(features, mask)
        features = K.conv2d(features,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        norm = K.conv2d(mask,
            self.mask_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        features = tf.math.divide_no_nan(features, norm)

        if self.use_bias:
            features = tf.add(features, self.bias)

        # Apply activations on the image
        if self.activation is not None:
            features = self.activation(features)

        mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)

        return [features, mask]

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

