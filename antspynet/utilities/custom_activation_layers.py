from tensorflow.keras.layers import Layer
import tensorflow as tf

class LogSoftmax(Layer):

    """Log Softmax activation function.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Arguments:
        axis: Integer, axis along which the softmax normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return tf.nn.log_softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(LogSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


