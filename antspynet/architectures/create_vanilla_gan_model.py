
import keras.backend as K

from keras.models import Model, Sequential
from keras.engine import Layer, InputSpec
from keras.layers import (Input, Concatenate, Dense, Activation,
                          BatchNormalization, Reshape,
                          Flatten, LeakyReLU)
from keras import optimizers

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class VanillaGanModel(object):
    """
    Deep embedded clustering with and without convolutions.

    Arguments
    ---------
    input_image_size : tuple
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    latent_dimension : integer

    Returns
    -------
    A keras vanilla GAN model.
    """

    def __init__(self, input_image_size, latent_dimension=100):
        super(VanillaGanModel, self).__init__()

        self.input_image_size = input_image_size
        self.latent_dimension = latent_dimension

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizers.adam(lr=0.0001), metrics=['acc'])
        self.discriminator.trainable = False

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dimension))
        image = self.generator(z)

        validity = self.discriminator(image)

        self.combined_model = Model(inputs=z, outputs=validity)
        self.combined_model.compile(loss = 'binary_crossentropy',
                                    optimizer=optimizers.adam(lr=0.0001))

    def build_generator(self):
        model = Sequential()

        for i in range(3):
            number_of_units = 2 ** (8 + i)

            if i == 0:
                model.add(Dense(input_shape=self.latent_dimension,
                                units=number_of_units))
            else:
                model.add(Dense(units=number_of_units))

            model.add(Dense(units=number_of_units))
            model.add(LeakyReLu(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))

        size = 1.0
        for i in range(len(self.input_image_size)):
            size *= self.input_image_size[i]

        model.add(Dense(units=size))
        model.add(Reshape(target_shape=self.input_image_size))

        noise = Input(shape=(self.latent_dimension))
        image = model(noise)

        generator = Model(inputs=noise, outputs=image)
        return(generator)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.input_image_size))
        model.add(Dense(units=512))
        model.add(LeakyReLu(alpha=0.2))
        model.add(Dense(units=1,
                        activation='sigmoid'))

        image = Input(shape=self.input_image_size)

        validity = model(image)

        discriminator = Model(inputs=image, outputs=validity)
        return(discriminator)

    def train(self, X_train, number_of_epochs, batch_size=128):
        valid = np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(number_of_epochs):

            # train discriminator

            indices = np.random.randint(0, X_train.shape[0], batch_size)
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




