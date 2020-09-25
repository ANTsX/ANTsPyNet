
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer, InputSpec

from tensorflow.keras import initializers

class Scale(Layer):

    """
    Custom layer used in the Dense U-net class for normalization which
    learns a set of weights and biases for scaling the input data.

    Arguments
    ---------

    axis : integer
        Specifies which axis to normalize.

    momentum : scalar
        Value used for computation of the exponential average of the
        mean and standard deviation.

    """

    def __init__(self, axis=-1, momentum=0.9, **kwargs):
        self.momentum = momentum
        self.axis = axis

        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        output_shape = (int(input_shape[self.axis]),)

        gamma_initializer = initializers.Ones()
        beta_initializer = initializers.Zeros()

        self.gamma = K.variable(gamma_initializer(output_shape))
        self.beta = K.variable(beta_initializer(output_shape))
        self.trainable_weights = [self.gamma, self.beta]

    def call(self, inputs, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        output = (K.reshape(self.gamma, broadcast_shape) * inputs +
                  K.reshape(self.beta, broadcast_shape))
        return(output)

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



