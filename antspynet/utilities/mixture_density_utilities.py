
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Concatenate
from tensorflow.keras import initializers

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class MixtureDensityLayer(Layer):

    """
    Layer for modeling arbitrary functions using neural networks.

    Arguments
    ---------
    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    Returns
    -------
    Layer
        A keras layer

    """

    def __init__(self, output_dimension, number_of_mixtures, **kwargs):
        if K.backend() != 'tensorflow':
            raise ValueError("Tensorflow required as the backend.")

        self.output_dimension = output_dimension
        self.number_of_mixtures = number_of_mixtures

        super(MixtureDensityLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        input_dimension = input_shape[-1]

        units1 = self.output_dimension * self.number_of_mixtures

        self.mu_kernel = self.add_weight(name="mu_kernel",
                                         shape = shape(input_dimension, units1),
                                         initializer=initializers.random_normal(),
                                         trainable=True)
        self.mu_bias = self.add_weight(name="mu_bias",
                                       shape = shape(units1),
                                       initializer=initializers.zeros(),
                                       trainable=True)

        self.sigma_kernel = self.add_weight(name="sigma_kernel",
                                            shape = shape(input_dimension, units1),
                                            initializer=initializers.random_normal(),
                                            trainable=True)
        self.sigma_bias = self.add_weight(name="sigma_bias",
                                          shape = shape(units1),
                                          initializer=initializers.zeros(),
                                          trainable=True)

        units2 = self.number_of_mixtures

        self.pi_kernel = self.add_weight(name="pi_kernel",
                                         shape = shape(input_dimension, units2),
                                         initializer=initializers.random_normal(),
                                         trainable=True)
        self.pi_bias = self.add_weight(name="pi_bias",
                                       shape = shape(units2),
                                       initializer=initializers.zeros(),
                                       trainable=True)

    def call(self, inputs, mask=None):

        # dense layer for mu (mean) of the gaussians
        mu_output = K.dot(inputs, self.mu_kernel)
        mu_output = K.bias_add(mu_output, self.mu_bias, data_format='channels_last')

        # dense layer for sigma (variance) of the gaussians
        sigma_output = K.dot(inputs, self.sigma_kernel)
        sigma_output = K.bias_add(sigma_output, self.sigma_bias, data_format='channels_last')

        # Avoid NaN's by pushing sigma through the following custom activation
        sigma_output = K.elu(sigma_output) + 1 + K.epsilon()

        # dense layer for pi (amplitude) of the gaussians
        pi_output = K.dot( inputs, self.pi_kernel)
        pi_output = K.bias_add(pi_output, self.pi_bias, data_format='channels_last')

        output = Concatenate()([mu_output, sigma_output, pi_output], name="mdn_outputs")
        return(output)

    def compute_output_shape(input_shape):
        units = self.number_of_mixtures * (2 * self.output_dimension + 1)
        return((input_shape[0], units))

    def get_config(self):
        config = {"output_dimension": self.output_dimension,
                  "axis": self.number_of_mixtures}
        base_config = super(MixtureDensityLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_mixture_density_loss_function(output_dimension, number_of_mixtures):

    """
    Returns a loss function for the mixture density.

    Arguments
    ---------
    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    Returns
    -------
    Function
        A function providing the mean square error accuracy

    """

    def loss_function(y_true, y_pred):

        dimension = number_of_mixtures * output_dimension

        y_pred = tf.reshape(y_pred, [-1, 2 * dimension + number_of_mixtures],
                            name='reshape_ypred_loss')
        y_true = tf.reshape(y_true, [-1, 2 * dimension + number_of_mixtures],
                            name='reshape_ytrue_loss')

        output_mu, output_sigma, output_pi = tf.split(y_pred, axis=-1, name='mdn_coef_split',
                   num_or_size_splits=[dimension, dimension, number_of_mixtures])

        # Construct the mixture models

        tfd = tfp.distributions

        categorical_distribution = tfd.Categorical(logits=output_pi)
        component_splits = [output_dimension] * number_of_mixtures
        mu = tf.split(output_mu, num_or_size_splits=component_splits, axis=1)
        sigma = tf.split(output_sigma, num_or_size_splits=component_splits, axis=1)

        components = []
        for i in range(len(mu)):
            components.append(tfd.MultivariateNormalDiag(loc = mu[i], scale_diag=sigma[i]))

        mixture = tfd.Mixture(cat=categorical_distribution, components=components)

        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)

        return(loss)

    with tf.name_scope("MixtureDensityNetwork"):
        return(loss_function)

def get_mixture_density_sampling_function(output_dimension, number_of_mixtures):

    """
    Returns a sampling function for the mixture density.

    Arguments
    ---------
    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    Returns
    -------
    Function
        A function providing the mean square error accuracy

    """

    def sampling_function(y_pred):

        dimension = number_of_mixtures * output_dimension

        y_pred = tf.reshape(y_pred, [-1, 2 * dimension + number_of_mixtures],
                            name='reshape_ypred')

        output_mu, output_sigma, output_pi = tf.split(y_pred, axis=-1, name='mdn_coef_split',
                   num_or_size_splits=[dimension, dimension, number_of_mixtures])

        # Construct the mixture models

        tfd = tfp.distributions

        categorical_distribution = tfd.Categorical(logits=output_pi)
        component_splits = [output_dimension] * number_of_mixtures
        mu = tf.split(output_mu, num_or_size_splits=component_splits, axis=1)
        sigma = tf.split(output_sigma, num_or_size_splits=component_splits, axis=1)

        components = []
        for i in range(len(mu)):
            components.append(tfd.MultivariateNormalDiag(loc = mu[i], scale_diag=sigma[i]))

        mixture = tfd.Mixture(cat=categorical_distribution, components=components)

        sample = mixture.sample()

        return(sample)

    with tf.name_scope("MixtureDensityNetwork"):
        return(sampling_function)


def get_mixture_density_mse_function(output_dimension, number_of_mixtures):

    """
    Returns a mse function for the mixture density.

    Arguments
    ---------
    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    Returns
    -------
    Function
        A function providing the mean square error accuracy

    """

    def mse_accuracy_function(y_true, y_pred):

        dimension = number_of_mixtures * output_dimension

        y_pred = tf.reshape(y_pred, [-1, 2 * dimension + number_of_mixtures],
                            name='reshape_ypred_mse')
        y_true = tf.reshape(y_true, [-1, output_dimension],
                            name='reshape_ytrue_mse')

        output_mu, output_sigma, output_pi = tf.split(y_pred, axis=-1, name='mdn_coef_split',
                   num_or_size_splits=[dimension, dimension, number_of_mixtures])

        # Construct the mixture models

        tfd = tfp.distributions

        categorical_distribution = tfd.Categorical(logits=output_pi)
        component_splits = [output_dimension] * number_of_mixtures
        mu = tf.split(output_mu, num_or_size_splits=component_splits, axis=1)
        sigma = tf.split(output_sigma, num_or_size_splits=component_splits, axis=1)

        components = []
        for i in range(len(mu)):
            components.append(tfd.MultivariateNormalDiag(loc = mu[i], scale_diag=sigma[i]))

        mixture = tfd.Mixture(cat=categorical_distribution, components=components)

        sample = mixture.sample()
        mse = tf.reduce_mean(tf.square(sample-y_true), axis=-1)

        return(mse)

    with tf.name_scope("MixtureDensityNetwork"):
        return(mse_accuracy_function)

def split_mixture_parameters(parameters, output_dimension, number_of_mixtures):

    """
    Splits the mixture parameters.

    Arguments
    ---------

    parameters : tuple
        Parameter to split

    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    Returns
    -------
    List of arrays
        Separate mixture parameters

    """

    dimension = number_of_mixtures * output_dimension
    mu = parameters[:dimension]
    sigma = parameters[dimension:(2 * dimension)]
    pi_logits = parameters[-number_of_mixtures:]
    return([mu, sigma, pi_logits])

def mixture_density_software_max(logits, temperature=1.0):

    """
    Softmax function for mixture density with temperature adjustment.

    Arguments
    ---------

    logits : list or numpy array
        input

    temperature :
        The temperature for to adjust the distribution (default 1.0)

    Returns
    -------
    Scalar
        Softmax loss value.

    """

    e = np.array(logits) / temperature
    e -= np.max(e)
    e = np.exp(e)

    distribution = e / np.sum(e)

    return(distribution)

def sample_from_categorical_distribution(distribution):

    """
    Softmax function for mixture density with temperature adjustment.

    Arguments
    ---------

    distribution :
        input categorical distribution from which to sample.

    Returns
    -------
    Scalar
        A single sample.

    """

    r = np.random.rand(1)

    accumulate = 0
    for i in range(len(distribution)):
        accumulate += distribution[i]
        if accumulate >= r:
            return(i)

    tf.logging.info('Error: sampling categorical model.')
    return(-1)

def sample_from_output(parameters, output_dimension, number_of_mixtures,
                       temperature=1.0, sigma_temperature=1.0):

    """
    Softmax function for mixture density with temperature adjustment.

    Arguments
    ---------
    output_dimension : integer
        Dimensionality of the output.

    number_of_mixtures : integer
        Number of gaussians used.

    temperature :
        The temperature for to adjust the distribution (default 1.0)

    sigma_temperature :
        The temperature for to adjust the distribution (default 1.0)

    Returns
    -------
    Scalar
        A single sample.

    """

    mu, sigma, pi = split_mixture_parameters(parameters, output_dimension, number_of_mixtures)
    pi_softmax = mixture_density_software_max(pi, temperature=temperature)
    m = sample_from_categorical_distribution(pi_softmax)

    mu_vector = mu[m * output_dimension:(m + 1) * output_dimension]
    sigma_vector = sigma[m * output_dimension:(m + 1) * output_dimension] * sigma_temperature
    covariance_matrix = np.identity(output_dimension) * sigma_vector
    sample = np.random.multivariate_normal(mu_vector, covariance_matrix, 1)
    return(sample)
