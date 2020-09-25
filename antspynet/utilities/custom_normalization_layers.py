import tensorflow as tf

from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import backend as K


class InstanceNormalization(Layer):

    """
    Instance normalization layer.

    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    Taken from

    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/normalization/instancenormalization.py

    Arguments
    ---------

    axis: integer
        Integer specifying which axis should be normalized, typically
        the feature axis.  For example, after a Conv2D layer with
        `channels_first`, set axis = 1.  Setting `axis=-1L` will
        normalize all values in each instance of the batch.  Axis 0
        is the batch dimension for tensorflow backend so we throw an
        error if `axis = 0`.

    epsilon: float
        Small float added to variance to avoid dividing by zero.

    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.

    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.  When the next layer is linear (also e.g.,
        `nn.relu`), this can be disabled since the scaling will be done by the
        next layer.

    beta_initializer : string
        Initializer for the beta weight.

    gamma_initializer : string
        Initializer for the gamma weight.

    beta_regularizer : string
        Optional regularizer for the beta weight.

    gamma_regularizer : string
        Optional regularizer for the gamma weight.

    beta_constraint : string
        Optional constraint for the beta weight.

    gamma_constraint : string
        Optional constraint for the gamma weight.

    Returns
    -------
    Keras layer


    """
    def __init__(self, axis=None, epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, **kwargs):

        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dimensionality = len(input_shape)

        if (self.axis is not None) and (dimensionality == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=dimensionality)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True)
        normed = (inputs - mean) / (stddev + self.epsilon)

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))