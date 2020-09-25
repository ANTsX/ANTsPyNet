import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Input, Concatenate, Dense, Activation,
                                     BatchNormalization, Reshape, Dropout,
                                     Flatten, LeakyReLU, Conv2D, Conv3D,
                                     UpSampling2D, UpSampling3D)
from tensorflow.keras.optimizers import Adam

from ..utilities import InstanceNormalization

import numpy as np
import os

import matplotlib.pyplot as plot

import ants

class CycleGanModel(object):
    """
    Cycle GAN model

    Cycle generative adverserial network from the paper:

      https://arxiv.org/pdf/1703.10593

    and ported from the Keras (python) implementation:

      https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py

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

    def __init__(self, input_image_size, lambda_cycle_loss_weight=10.0,
                 lambda_identity_loss_weight=1.0,
                 number_of_filters_at_base_layer=(32, 64)):
        super(CycleGanModel, self).__init__()

        self.input_image_size = input_image_size
        self.number_of_channels = self.input_image_size[-1]
        self.discriminator_patch_size = None
        self.lambda_cycle_loss_weight = lambda_cycle_loss_weight
        self.lambda_identity_loss_weight = lambda_identity_loss_weight

        self.number_of_filters_at_base_layer = number_of_filters_at_base_layer

        self.dimensionality = None
        if len(self.input_image_size) == 3:
            self.dimensionality = 2
        elif len(self.input_image_size) == 4:
            self.dimensionality = 3
        else:
            raise ValueError("Incorrect size for input_image_size.")

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Build discriminators for domains A and B

        self.discriminatorA = self.build_discriminator()
        self.discriminatorA.compile(loss='mse',
                                   optimizer=optimizer, metrics=['acc'])
        self.discriminatorA.trainable = False

        self.discriminatorB = self.build_discriminator()
        self.discriminatorB.compile(loss='mse',
                                   optimizer=optimizer, metrics=['acc'])
        self.discriminatorB.trainable = False

        # Build u-net like generators

        self.generatorAtoB = self.build_generator()
        self.generatorBtoA = self.build_generator()

        imageA = Input(shape=input_image_size)
        imageB = Input(shape=input_image_size)

        fake_imageA = self.generatorBtoA(imageB)
        fake_imageB = self.generatorAtoB(imageA)

        reconstructed_imageA = self.generatorBtoA(fake_imageB)
        reconstructed_imageB = self.generatorAtoB(fake_imageA)

        identity_imageA = self.generatorBtoA(imageA)
        identity_imageB = self.generatorAtoB(imageB)

        # Check images

        validityA = self.discriminatorA(fake_imageA)
        validityB = self.discriminatorB(fake_imageB)

        # Combined models

        self.combined_model = Model(inputs=[imageA, imageB],
                                    outputs=[validityA, validityB,
                                             reconstructed_imageA, reconstructed_imageB,
                                             identity_imageA, identity_imageB])
        self.combined_model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                                    loss_weights=[1.0, 1.0,
                                                  self.lambda_cycle_loss_weight, self.lambda_cycle_loss_weight,
                                                  self.lambda_identity_loss_weight, self.lambda_identity_loss_weight],
                                    optimizer=optimizer)

    def build_generator(self):

        def build_encoding_layer(input, number_of_filters, kernel_size=4):
            encoder = input
            if self.dimensionality == 2:
                encoder = Conv2D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(encoder)
            else:
                encoder = Conv3D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(encoder)
            encoder = LeakyReLU(alpha=0.2)(encoder)
            encoder = InstanceNormalization()(encoder)
            return(encoder)

        def build_decoding_layer(input, skip_input, number_of_filters,
                                 kernel_size=4, dropout_rate=0.0):
            decoder = input
            if self.dimensionality == 2:
                decoder = UpSampling2D(size=2)(decoder)
                decoder = Conv2D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='relu')(decoder)
            else:
                decoder = UpSampling3D(size=2)(decoder)
                decoder = Conv3D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='relu')(decoder)
            if dropout_rate > 0.0:
                decoder = Dropout(dropout_rate=dropout_rate)(decoder)
            decoder = LeakyReLU(alpha=0.2)(decoder)
            decoder = Concatenate()([decoder, skip_input])
            return(decoder)

        input = Input(shape=self.input_image_size)

        encoding_layers = list()
        encoding_layers.append(build_encoding_layer(input,
                                                    int(self.number_of_filters_at_base_layer[0])))
        encoding_layers.append(build_encoding_layer(encoding_layers[0],
                                                    int(self.number_of_filters_at_base_layer[0] * 2)))
        encoding_layers.append(build_encoding_layer(encoding_layers[1],
                                                    int(self.number_of_filters_at_base_layer[0] * 4)))
        encoding_layers.append(build_encoding_layer(encoding_layers[2],
                                                    int(self.number_of_filters_at_base_layer[0] * 8)))

        decoding_layers = list()
        decoding_layers.append(build_decoding_layer(encoding_layers[3], encoding_layers[2],
                                                    int(self.number_of_filters_at_base_layer[0] * 4)))
        decoding_layers.append(build_decoding_layer(decoding_layers[0], encoding_layers[1],
                                                    int(self.number_of_filters_at_base_layer[0] * 2)))
        decoding_layers.append(build_decoding_layer(decoding_layers[1], encoding_layers[0],
                                                    int(self.number_of_filters_at_base_layer[0])))

        if self.dimensionality == 2:
            decoding_layers.append(UpSampling2D(size=2)(decoding_layers[-1]))
            decoding_layers[-1] = Conv2D(filters=self.number_of_channels,
                                         kernel_size=4,
                                         strides=1,
                                         padding='same',
                                         activation='tanh')(decoding_layers[-1])
        else:
            decoding_layers.append(UpSampling3D(size=2)(decoding_layers[-1]))
            decoding_layers[-1] = Conv2D(filters=self.number_of_channels,
                                         kernel_size=4,
                                         strides=1,
                                         padding='same',
                                         activation='tanh')(decoding_layers[-1])

        generator = Model(inputs=input, outputs=decoding_layers[-1])
        return(generator)

    def build_discriminator(self):

        def build_layer(input, number_of_filters, kernel_size=4, normalization=True):
            layer = input
            if self.dimensionality == 2:
                layer = Conv2D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(layer)
            else:
                layer = Conv3D(filters=number_of_filters,
                                 kernel_size=kernel_size,
                                 strides=2,
                                 padding='same')(layer)
            layer = LeakyReLU(alpha=0.2)(layer)
            if normalization == True:
                layer = InstanceNormalization()(layer)
            return(layer)

        input = Input(shape=self.input_image_size)

        layers = list()
        layers.append(build_layer(input,
                                  int(self.number_of_filters_at_base_layer[1])))
        layers.append(build_layer(layers[0],
                                  int(self.number_of_filters_at_base_layer[1] * 2)))
        layers.append(build_layer(layers[1],
                                  int(self.number_of_filters_at_base_layer[1] * 4)))
        layers.append(build_layer(layers[2],
                                  int(self.number_of_filters_at_base_layer[1] * 8)))

        validity = None
        if self.dimensionality == 2:
            validity = Conv2D(filters=1,
                              kernel_size=4,
                              strides=1,
                              padding='same')(layers[3])
        else:
            validity = Conv3D(filters=1,
                              kernel_size=4,
                              strides=1,
                              padding='same')(layers[3])

        if self.discriminator_patch_size is None:
            self.discriminator_patch_size = K.int_shape(validity)[1:]

        discriminator = Model(inputs=input, outputs=validity)
        return(discriminator)

    def train(self, X_trainA, X_trainB, number_of_epochs, batch_size=128,
              sample_interval=None, sample_file_prefix='sample'):

        valid = np.ones((batch_size, *self.discriminator_patch_size))
        fake = np.zeros((batch_size, *self.discriminator_patch_size))

        for epoch in range(number_of_epochs):

            indicesA = np.random.randint(0, X_trainA.shape[0] - 1, batch_size)
            imagesA = X_trainA[indicesA]

            indicesB = np.random.randint(0, X_trainB.shape[0] - 1, batch_size)
            imagesB = X_trainB[indicesB]

            # train discriminator

            fake_imagesA = self.generatorAtoB.predict(imagesA)
            fake_imagesB = self.generatorBtoA.predict(imagesB)

            dA_loss_real = self.discriminatorA.train_on_batch(imagesA, valid)
            dA_loss_fake = self.discriminatorA.train_on_batch(fake_imagesA, fake)

            dB_loss_real = self.discriminatorB.train_on_batch(imagesB, valid)
            dB_loss_fake = self.discriminatorB.train_on_batch(fake_imagesB, fake)

            d_loss = list()
            for i in range(len(dA_loss_real)):
                d_loss.append(0.25 * (dA_loss_real[i] + dA_loss_fake[i] +
                                      dB_loss_real[i] + dB_loss_fake[i]))

            # train generator

            g_loss = self.combined_model.train_on_batch([imagesA, imagesB],
              [valid, valid, imagesA, imagesB, imagesA, imagesB])

            print("Epoch ", epoch, ": [Discriminator loss: ", d_loss[0],
                  " acc: ", d_loss[1], "] ", "[Generator loss: ", g_loss[0],
                  ", ", np.mean(g_loss[1:3]), ", ", np.mean(g_loss[3:5]), ", ",
                  np.mean(g_loss[5:6]), "]")

            if self.dimensionality == 2:
                if sample_interval != None:
                    if epoch % sample_interval == 0:

                        # Do a 2x3 grid
                        #
                        # imageA  |  translated( imageA ) | reconstructed( imageA )
                        # imageB  |  translated( imageB ) | reconstructed( imageB )

                        indexA = np.random.randint(0, X_trainA.shape[0] - 1, 1)
                        indexB = np.random.randint(0, X_trainB.shape[0] - 1, 1)

                        imageA = X_trainA[indexA,:,:,:]
                        imageB = X_trainB[indexB,:,:,:]

                        X = list()
                        X.append(imageA)
                        X.append(self.generatorAtoB.predict(X[0]))
                        X.append(self.generatorBtoA.predict(X[1]))

                        X.append(imageB)
                        X.append(self.generatorAtoB.predict(X[3]))
                        X.append(self.generatorBtoA.predict(X[4]))

                        plot_images = np.concatenate(X)
                        plot_images = 0.5 * plot_images + 0.5

                        titles = ['Original', 'Translated', 'Reconstructed']
                        figure, axes = plot.subplots(2, 3)

                        count = 0
                        for i in range(2):
                            for j in range(3):
                                axes[i, j].imshow(plot_images[count])
                                axes[i, j].set_title(titles[j])
                                axes[i, j].axis('off')
                                count += 1

                        image_file_name = sample_file_prefix + "_iteration" + str(epoch) + ".jpg"
                        dir_name = os.path.dirname(sample_file_prefix)
                        if not os.path.exists(dir_name):
                            os.mkdir(dir_name)
                        figure.savefig(image_file_name)
                        plot.close()

