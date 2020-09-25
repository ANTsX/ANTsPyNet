
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Input, Concatenate, Dense, Activation,
                                     BatchNormalization, Reshape, Dropout,
                                     Flatten, LeakyReLU, Conv2D, Conv3D)
from tensorflow.keras.optimizers import Adam

from . import (create_convolutional_autoencoder_model_2d,
               create_convolutional_autoencoder_model_3d)

from ..utilities import ResampleTensorLayer2D, ResampleTensorLayer3D

import numpy as np
import os

import ants

class DeepConvolutionalGanModel(object):
    """
    GAN model using CNNs

    Deep convolutional generative adverserial network from the paper:

      https://arxiv.org/abs/1511.06434

    and ported from the Keras (python) implementation:

      https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py

    Arguments
    ---------
    input_image_size : tuple
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    latent_dimension : integer

    Returns
    -------
    Keras model
        A Keras model defining the network.
    """

    def __init__(self, input_image_size, latent_dimension=100):
        super(DeepConvolutionalGanModel, self).__init__()

        self.input_image_size = input_image_size
        self.latent_dimension = latent_dimension

        self.dimensionality = None
        if len(self.input_image_size) == 3:
            self.dimensionality = 2
        elif len(self.input_image_size) == 4:
            self.dimensionality = 3
        else:
            raise ValueError("Incorrect size for input_image_size.")

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer, metrics=['acc'])
        self.discriminator.trainable = False

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dimension,))
        image = self.generator(z)

        validity = self.discriminator(image)

        self.combined_model = Model(inputs=z, outputs=validity)
        self.combined_model.compile(loss='binary_crossentropy',
                                    optimizer=optimizer)

    def build_generator(self, number_of_filters_per_layer=(128, 64), kernel_size=3):

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

    def build_discriminator(self, number_of_filters_per_layer=(32, 64, 128, 256),
                            kernel_size=3, dropout_rate=0.25):
        model = Sequential()

        for i in range(len(number_of_filters_per_layer)):
            if self.dimensionality == 2:
                model.add(Conv2D(input_shape=self.input_image_size,
                                 filters=number_of_filters_per_layer[i],
                                 kernel_size = kernel_size,
                                 strides = 2,
                                 padding='same'))
            else:
                model.add(Conv3D(input_shape=self.input_image_size,
                                 filters=number_of_filters_per_layer[i],
                                 kernel_size = kernel_size,
                                 strides = 2,
                                 padding='same'))

            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(rate=dropout_rate))

        model.add(Flatten())
        model.add(Dense(units=1, activation='sigmoid'))

        image = Input(shape=self.input_image_size)

        validity = model(image)

        discriminator = Model(inputs=image, outputs=validity)

        return(discriminator)

    def train(self, X_train, number_of_epochs, batch_size=128,
              sample_interval=None, sample_file_prefix='sample'):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(number_of_epochs):

            # train discriminator

            indices = np.random.randint(0, X_train.shape[0] - 1, batch_size)
            X_valid_batch = X_train[indices]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dimension))
            X_fake_batch = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(X_valid_batch, valid)
            d_loss_fake = self.discriminator.train_on_batch(X_fake_batch, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator

            noise = np.random.normal(0, 1, (batch_size, self.latent_dimension))
            g_loss = self.combined_model.train_on_batch(noise, valid)

            print("Epoch ", epoch, ": [Discriminator loss: ", d_loss[0],
                  " acc: ", d_loss[1], "] ", "[Generator loss: ", g_loss)

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




