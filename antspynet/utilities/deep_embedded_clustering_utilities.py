
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers

import numpy as np

from ..architectures import create_autoencoder_model 
from ..architectures import create_convolutional_autoencoder_model_2d
from ..architectures import create_convolutional_autoencoder_model_3d

from sklearn.cluster import KMeans

class DeepEmbeddedClustering(Layer):

    """
    Deep embedded lustering layer.

    Arguments
    ---------
    number_of_clusters : integer
        Specifies which axis to normalize.

    initial_cluster_weights : list
        Initial clustering weights.

    alpha : scalar
        Parameter.

    Returns
    -------
    Keras layer
        A keras layer

    """

    def __init__(self, number_of_clusters=10, initial_cluster_weights=None, alpha=1.0, **kwargs):
        super(DeepEmbeddedClustering, self).__init__(**kwargs)

        self.number_of_clusters = number_of_clusters
        self.initial_cluster_weights = initial_cluster_weights
        self.alpha = alpha
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_shape[1]))

        if len(input_shape) != 2:
            raise ValueError("input_shape is not of length 2.")

        self.clusters = self.add_weight(shape=(self.number_of_clusters, input_shape[1]),
                                        initializer=initializers.glorot_uniform(),
                                        name='clusters')

        if self.initial_cluster_weights != None:
            self.set_weights(self.initial_cluster_weights)
            self.initial_cluster_weights = None

        self.built = True

    def call(self, inputs, mask=None):
        # Uses Student t-distribution (same as t-SNE)
        # inputs are the variable containing the data, shape=(number_of_samples, number_of_features)

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis = 1)
            - self.clusters), axis=2) / self.alpha))
        q **= ((self.alpha + 1.0) / 2.0)
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return(q)

    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.number_of_clusters)

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(DeepEmbeddedClustering, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DeepEmbeddedClusteringModel(object):
    """
    Deep embedded clustering with and without convolutions.

    Arguments
    ---------

    number_of_units_per_layer : integer
        Autoencoder number of units per layer.

    number_of_clusters : integer
        Number of clusters.

    alpha : scalar
        Parameter

    initializer : string
        Initializer for autoencoder.

    Returns
    -------
    Keras model
        A keras clustering model.

    """

    def __init__(self, number_of_units_per_layer=None, number_of_clusters=10, alpha=1.0,
                 initializer='glorot_uniform', convolutional=False, input_image_size=None):

        super(DeepEmbeddedClusteringModel, self).__init__()

        self.number_of_units_per_layer = number_of_units_per_layer
        self.number_of_clusters = number_of_clusters
        self.alpha = alpha
        self.initializer = initializer
        self.convolutional = convolutional
        self.input_image_size = input_image_size
        self.autoencoder = None
        self.encoder = None

        if self.convolutional == True:
            if self.input_image_size == None:
                raise ValueError("Need to specify the input image size for CNN.")

            if len(self.input_image_size) == 3:  # 2-D
                self.autoencoder, self.encoder = create_convolutional_autoencoder_model_2d(
                       input_image_size=self.input_image_size,
                       number_of_filters_per_layer=self.number_of_units_per_layer)
            else:
                self.autoencoder, self.encoder = create_convolutional_autoencoder_model_3d(
                       input_image_size=self.input_image_size,
                       number_of_filters_per_layer=self.number_of_units_per_layer)

        else:
            self.autoencoder, self.encoder = create_autoencoder_model(
                self.number_of_units_per_layer, initializer=self.initializer)

        clustering_layer = DeepEmbeddedClustering(
            self.number_of_clusters, name="clustering")(self.encoder.output) 

        if self.convolutional == True:
            self.model = Model(inputs=self.encoder.input, outputs=[clustering_layer, self.autoencoder.output])
        else:
            self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256):
        self.autoencoder.compile(optimizer=optimizer, loss='mse')
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        self.encoder.predict(x, verbose=0)

    def predict_cluster_labels(self, x):
        cluster_probabilities = self.model.predict(x, verbose=0)
        return(cluster_probabilities.argmax(1))

    def target_distribution(self, q):
        weight = q**2 / q.sum(0)
        p = (weight.T / weight.sum(1)).T
        return(p)

    def compile(self, optimizer='sgd', loss='kld', loss_weights=None):
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    def fit(self, x, max_number_of_iterations=2e4, batch_size=256, tolerance=1e-3, update_interval=140):

        # Initialize clusters using k-means

        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=20)
        current_prediction = kmeans.fit_predict(self.encoder.predict(x))
        previous_prediction = np.copy(current_prediction)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Deep clustering

        loss = 100000000
        index = 0
        index_array = np.arange(x.shape[0])

        for i in range(max_number_of_iterations):

            if i % update_interval == 0:
                if self.convolutional == True:
                    q, _ = self.model.predict(x, verbose=0)
                else:
                    q = self.model.predict(x, verbose=0)                    
                p = self.target_distribution(q)

                # Met stopping criterion

                current_prediction = q.argmax(1)
                delta_label = (np.sum(current_prediction != previous_prediction).astype(np.float32) /
                               current_prediction.shape[0])
                previous_prediction = np.copy(current_prediction)

                print("Iteration ", i, " (out of ", max_number_of_iterations, 
                  "): loss = ", loss, ", delta_label = ", delta_label)

                if i > 0 and delta_label < tolerance:
                    break

            batch_indices = index_array[index * batch_size:min((index + 1) * batch_size, x.shape[0])]

            if self.convolutional == True:
                if len(self.input_image_size) == 3:
                    loss = self.model.train_on_batch(x=x[batch_indices,:,:,:], 
                                                     y=[p[batch_indices,:], x[batch_indices,:,:,:]])
                else:
                    loss = self.model.train_on_batch(x=x[batch_indices,:,:,:,:], 
                                                     y=[p[batch_indices,:,:], x[batch_indices,:,:,:,:]])
            else:
                loss = self.model.train_on_batch(x=x[batch_indices,:], y=p[batch_indices,:])

            if (index + 1) * batch_size <= x.shape[0]:
                index += 1
            else:
                index = 0

        return(current_prediction)
