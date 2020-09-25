import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Input, Concatenate, Dense, Activation,
                                     BatchNormalization, Reshape, Dropout,
                                     Flatten, LeakyReLU, Conv2D, Conv3D)
from tensorflow.keras.optimizers import RMSprop

from . import (create_convolutional_autoencoder_model_2d,
               create_convolutional_autoencoder_model_3d)

from ..utilities import ResampleTensorLayer2D, ResampleTensorLayer3D

import numpy as np
import os

import ants

class WassersteinGanModel(object):
    """
    Wasserstein GAN model

    Wasserstein generative adverserial network from the paper:

      https://arxiv.org/abs/1701.07875

    and ported from the Keras implementation:

      https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py

    Arguments
    ---------
    input_image_size : tuple
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    latent_dimension : integer
        Default = 100.

    number_of_critic_iterations : integer
        Default = 5.

    clip_value : float
        Default = 0.01.

    Returns
    -------
    Keras model
        A Keras model defining the network.
    """

    def __init__(self, input_image_size, latent_dimension=100,
                 number_of_critic_iterations=5, clip_value=0.01):
        super(WassersteinGanModel, self).__init__()

        self.input_image_size = input_image_size
        self.latent_dimension = latent_dimension
        self.number_of_critic_iterations = number_of_critic_iterations
        self.clip_value = clip_value

        self.dimensionality = None
        if len(self.input_image_size) == 3:
            self.dimensionality = 2
        elif len(self.input_image_size) == 4:
            self.dimensionality = 3
        else:
            raise ValueError("Incorrect size for input_image_size.")

        optimizer = RMSprop(lr=0.00005)

        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer,
                            metrics=['acc'])
        self.critic.trainable = False

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dimension,))
        image = self.generator(z)

        validity = self.critic(image)

        self.combined_model = Model(inputs=z, outputs=validity)
        self.combined_model.compile(loss=self.wasserstein_loss,
                                    optimizer=optimizer, metrics=['acc'])

    def wasserstein_loss(self, y_true, y_pred):
       return(K.mean(y_true * y_pred))

    def build_generator(self, number_of_filters_per_layer=(128, 64), kernel_size=4):

        model = Sequential()

        # To build the generator, we create the reverse encoder model
        # and simply build the reverse model

        encoder = None
        if self.dimensionality == 2:
             autoencoder, encoder = create_convolutional_autoencoder_model_2d(
                          input_image_size=self.input_image_size,
                          number_of_filters_per_layer=(*(number_of_filters_per_layer[::-1]), self.latent_dimension),
                          convolution_kernel_size=(5, 5),
                          deconvolution_kernel_size=(5, 5))
        else:
             autoencoder, encoder = create_convolutional_autoencoder_model_3d(
                          input_image_size=self.input_image_size,
                          number_of_filters_per_layer=(*(number_of_filters_per_layer[::-1]), self.latent_dimension),
                          convolution_kernel_size=(5, 5, 5),
                          deconvolution_kernel_size=(5, 5, 5))

        encoder_layers = encoder.layers

        penultimate_layer = encoder_layers[len(encoder_layers) - 2]

        model.add(Dense(units=penultimate_layer.output_shape[1],
                        input_dim=self.latent_dimension,
                        activation="relu"))

        conv_layer = encoder_layers[len(encoder_layers) - 3]
        resampled_size = conv_layer.output_shape[1:(self.dimensionality + 2)]
        model.add(Reshape(resampled_size))

        count = 0
        for i in range(len(encoder_layers) - 3, 1, -1):
            conv_layer = encoder_layers[i]
            resampled_size = conv_layer.output_shape[1:(self.dimensionality + 1)]

            if self.dimensionality == 2:
                model.add(ResampleTensorLayer2D(shape=resampled_size,
                                                interpolation_type='linear'))
                model.add(Conv2D(filters=number_of_filters_per_layer[count],
                                 kernel_size=kernel_size,
                                 padding='same'))
            else:
                model.add(ResampleTensorLayer3D(shape=resampled_size,
                                                interpolation_type='linear'))
                model.add(Conv3D(filters=number_of_filters_per_layer[count],
                                 kernel_size=kernel_size,
                                 padding='same'))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation(activation='relu'))
            count += 1

        number_of_channels = self.input_image_size[-1]
        spatial_dimensions = self.input_image_size[:self.dimensionality]

        if self.dimensionality == 2:
            model.add(ResampleTensorLayer2D(shape=spatial_dimensions,
                                            interpolation_type='linear'))
            model.add(Conv2D(filters=number_of_channels,
                             kernel_size=kernel_size,
                             padding='same'))
        else:
            model.add(ResampleTensorLayer3D(shape=spatial_dimensions,
                                            interpolation_type='linear'))
            model.add(Conv3D(filters=number_of_channels,
                             kernel_size=kernel_size,
                             padding='same'))

        model.add(Activation(activation="tanh"))

        noise = Input(shape=(self.latent_dimension,))
        image = model(noise)

        generator = Model(inputs=noise, outputs=image)
        return(generator)

    def build_critic(self, number_of_filters_per_layer=(16, 32, 64, 128),
                            kernel_size=3, dropout_rate=0.25):
        model = Sequential()

        for i in range(len(number_of_filters_per_layer)):

            strides = 2
            if i == len(number_of_filters_per_layer) - 1:
                strides=1

            if self.dimensionality == 2:
                model.add(Conv2D(input_shape=self.input_image_size,
                                 filters=number_of_filters_per_layer[i],
                                 kernel_size = kernel_size,
                                 strides = strides,
                                 padding='same'))
            else:
                model.add(Conv3D(input_shape=self.input_image_size,
                                 filters=number_of_filters_per_layer[i],
                                 kernel_size = kernel_size,
                                 strides = strides,
                                 padding='same'))

            if i > 0:
                model.add(BatchNormalization(momentum=0.8))

            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(rate=dropout_rate))

        model.add(Flatten())
        model.add(Dense(units=1))

        image = Input(shape=self.input_image_size)

        validity = model(image)

        critic = Model(inputs=image, outputs=validity)

        return(critic)

    def train(self, X_train, number_of_epochs, batch_size=128,
              sample_interval=None, sample_file_prefix='sample'):
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(number_of_epochs):

            # train critic

            for c in range(self.number_of_critic_iterations):
                indices = np.random.randint(0, X_train.shape[0] - 1, batch_size)
                X_valid_batch = X_train[indices]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dimension))
                X_fake_batch = self.generator.predict(noise)

                d_loss_real = self.critic.train_on_batch(X_valid_batch, valid)
                d_loss_fake = self.critic.train_on_batch(X_fake_batch, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # clip critic weights

                for i in range(len(self.critic.layers)):
                    weights = self.critic.layers[i].get_weights()
                    for j in range(len(weights)):
                       weights[j] = np.clip(weights[j], -self.clip_value, self.clip_value)
                    self.critic.layers[i].set_weights(weights)

            # train generator

            noise = np.random.normal(0, 1, (batch_size, self.latent_dimension))
            g_loss = self.combined_model.train_on_batch(noise, valid)

            print("Epoch ", epoch, ": [Critic loss: ", 1.0 - d_loss[0],
                  "] ", "[Generator loss: ", 1.0 - g_loss[0])

            if self.dimensionality == 2:
                if sample_interval != None:
                    if epoch % sample_interval == 0:

                        # Do a 5x5 grid

                        predicted_batch_size = 5 * 5
                        noise = np.random.normal(0, 1, (predicted_batch_size, self.latent_dimension))
                        X_generated = self.generator.predict(noise)

                        # Convert to [0,255] to write as jpg using ANTsPy

                        X_generated = (255 * (X_generated - X_generated.min()) /
                          (X_generated.max() - X_generated.min()))
                        X_generated = np.squeeze(X_generated)
                        X_generated = np.uint8(X_generated)

                        X_tiled = np.zeros((5 * X_generated.shape[1], 5 * X_generated.shape[2]), dtype=np.uint8)
                        for i in range(5):
                            indices_i = (i * X_generated.shape[1], (i + 1) * X_generated.shape[1])
                            for j in range(5):
                                indices_j = (j * X_generated.shape[2], (j + 1) * X_generated.shape[2])
                                X_tiled[indices_i[0]:indices_i[1], indices_j[0]:indices_j[1]] = \
                                  np.squeeze(X_generated[i * 5 + j, :, :])

                        X_generated_image = ants.from_numpy(np.transpose(X_tiled))

                        image_file_name = sample_file_prefix + "_iteration" + str(epoch) + ".jpg"
                        dir_name = os.path.dirname(sample_file_prefix)
                        if not os.path.exists(dir_name):
                            os.mkdir(dir_name)

                        print("   --> writing sample image: ", image_file_name)
                        ants.image_write(X_generated_image, image_file_name)




