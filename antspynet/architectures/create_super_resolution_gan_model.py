
import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Input, Add, BatchNormalization,
                                     Conv2D, Conv3D, Dense, ReLU, LeakyReLU,
                                     UpSampling2D, UpSampling3D)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import vgg19

from . import create_vgg_model_2d, create_vgg_model_3d

import numpy as np
import os

import matplotlib.pyplot as plot

import ants

class SuperResolutionGanModel(object):
    """
    Super resolution GAN model

    Super resolution generative adverserial network from the paper:

      https://arxiv.org/abs/1609.04802

    and ported from the Keras implementation:

      https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

    Arguments
    ---------
    input_image_size : tuple
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    low_resolution_image_size : tuple
        Size of the input image.

    scale_factor : integer
        Upsampling factor for the output super-resolution image.

    use_image_net_weights : boolean
        Determines whether or not one uses the image-net weights.  Only valid for
        2-D images.

    number_of_residual_blocks : 16
        Number of residual blocks used in constructing the generator.

    number_of_filters_at_base_layer : tuple of length 2
        Number of filters at the base layer for the generator and discriminator,
        respectively.

    Returns
    -------
    Keras model
        A Keras model defining the network.
    """

    def __init__(self, low_resolution_image_size, scale_factor=2,
                 use_image_net_weights=True, number_of_residual_blocks=16,
                 number_of_filters_at_base_layer=(64, 64)):
        super(SuperResolutionGanModel, self).__init__()

        self.low_resolution_image_size = low_resolution_image_size
        self.number_of_channels = self.low_resolution_image_size[-1]
        self.number_of_residual_blocks = number_of_residual_blocks
        self.number_of_filters_at_base_layer = number_of_filters_at_base_layer
        self.use_image_net_weights = use_image_net_weights

        self.scale_factor = scale_factor
        if not self.scale_factor in set([1, 2, 4, 8]):
            raise ValueError("Scale factor must be one of 1, 2, 4, or 8.")

        self.dimensionality = None
        if len(self.low_resolution_image_size) == 3:
            self.dimensionality = 2
        elif len(self.low_resolution_image_size) == 4:
            self.dimensionality = 3
            if self.use_image_net_weights == True:
                self.use_image_net_weights = False
            print("Warning:  imageNet weights are unavailable for 3D.")
        else:
            raise ValueError("Incorrect size for low_resolution_image_size.")

        optimizer = Adam(lr=0.0002, beta_1=0.5)

        # Images

        tmp = list(self.low_resolution_image_size)
        for i in range(self.dimensionality):
            tmp[i] *= self.scale_factor
        self.high_resolution_image_size = tuple(tmp)

        high_resolution_image = Input(shape=self.high_resolution_image_size)

        low_resolution_image = Input(shape=self.low_resolution_image_size)

        # Build generator

        self.generator = self.build_generator()

        fake_high_resolution_image = self.generator(low_resolution_image)

        # Build discriminator

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer, metrics=['acc'])

        # Vgg

        self.vgg_model = self.build_truncated_vgg_model()
        self.vgg_model.trainable = False
        self.vgg_model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])

        if self.dimensionality == 2:
            self.discriminator_patch_size = (16, 16, 1)
        else:
            self.discriminator_patch_size = (16, 16, 16, 1)

        # Discriminator

        self.discriminator.trainable = False

        validity = self.discriminator(fake_high_resolution_image)

        # Combined model

        if self.use_image_net_weights == True:
            fake_features = self.vgg_model(fake_high_resolution_image)
            self.combined_model = Model(inputs=[low_resolution_image, high_resolution_image],
                                        outputs=[validity, fake_features])
            self.combined_model.compile(loss=['binary_crossentropy', 'mse'],
              loss_weights=[1e-3, 1], optimizer=optimizer)
        else:
            self.combined_model = Model(inputs=[low_resolution_image, high_resolution_image],
                                        outputs=validity)
            self.combined_model.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    def build_truncated_vgg_model(self):
        vgg_tmp = None
        if self.dimensionality == 2:
            if self.use_image_net_weights == True:
                vgg_tmp = create_vgg_model_2d((224, 224, 3), style=19)
                keras_vgg = vgg19(weights='imagenet')
                vgg_tmp.set_weights(keras_vgg.get_weights())
            else:
                vgg_tmp = create_vgg_model_2d(
                  self.high_resolution_image_size, style=19)
        else:
            vgg_tmp = create_vgg_model_3d(self.high_resolution_image_size, style=19)

        vgg_tmp.outputs = [vgg_tmp.layers[9].output]

        high_resolution_image = Input(shape=self.high_resolution_image_size)
        high_resolution_image_features = vgg_tmp(high_resolution_image)

        vgg_model = Model(inputs=high_resolution_image,
                          outputs=high_resolution_image_features)
        return(vgg_model)

    def build_generator(self, number_of_filters=64):

        def build_residual_block(input, number_of_filters, kernel_size=3):
           shortcut = input
           if self.dimensionality == 2:
               input = Conv2D(filters=number_of_filters,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same')(input)
           else:
               input = Conv3D(filters=number_of_filters,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same')(input)
           input = ReLU()(input)
           input = BatchNormalization(momentum=0.8)(input)
           if self.dimensionality == 2:
               input = Conv2D(filters=number_of_filters,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same')(input)
           else:
               input = Conv3D(filters=number_of_filters,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same')(input)
           input = BatchNormalization(momentum=0.8)(input)
           input = Add()([input, shortcut])
           return(input)

        def build_deconvolution_layer(input, number_of_filters=256, kernel_size=3):
            model = input
            if self.dimensionality == 2:
                model = UpSampling2D(size=2)(model)
                input = Conv2D(filters=number_of_filters,
                               kernel_size=kernel_size,
                               strides=1,
                               padding='same')(model)
            else:
                model = UpSampling3D(size=2)(model)
                input = Conv3D(filters=number_of_filters,
                               kernel_size=kernel_size,
                               strides=1,
                               padding='same')(model)
            model = ReLU()(model)
            return(model)

        image = Input(shape=self.low_resolution_image_size)

        pre_residual = image
        if self.dimensionality == 2:
            pre_residual = Conv2D(filters=number_of_filters,
                                  kernel_size=9,
                                  strides=1,
                                  padding='same')(pre_residual)
        else:
            pre_residual = Conv3D(filters=number_of_filters,
                                  kernel_size=9,
                                  strides=1,
                                  padding='same')(pre_residual)

        residuals = build_residual_block(pre_residual,
          number_of_filters=self.number_of_filters_at_base_layer[0])
        for i in range(self.number_of_residual_blocks - 1):
            residuals = build_residual_block(residuals,
              number_of_filters=self.number_of_filters_at_base_layer[0])

        post_residual = residuals
        if self.dimensionality == 2:
            post_residual = Conv2D(filters=number_of_filters,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same')(post_residual)
        else:
            post_residual = Conv3D(filters=number_of_filters,
                                  kernel_size=3,
                                  strides=1,
                                  padding='same')(post_residual)
        post_residual = BatchNormalization(momentum=0.8)(post_residual)
        model = Add()([post_residual, pre_residual])

        # upsampling

        if self.scale_factor >= 2:
            model = build_deconvolution_layer(model)
        if self.scale_factor >= 4:
            model = build_deconvolution_layer(model)
        if self.scale_factor == 8:
            model = build_deconvolution_layer(model)

        if self.dimensionality == 2:
            model = Conv2D(filters=self.number_of_channels,
                           kernel_size=9,
                           strides=1,
                           padding='same',
                           activation='tanh')(model)
        else:
            model = Conv3D(filters=self.number_of_channels,
                           kernel_size=9,
                           strides=1,
                           padding='same',
                           activation='tanh')(model)

        generator = Model(inputs=image, outputs=model)
        return(generator)

    def build_discriminator(self):

        def build_layer(input, number_of_filters, strides=1, kernel_size=3,
                        normalization=True):
            layer = input
            if self.dimensionality == 2:
                layer = Conv2D(filters=number_of_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same')(layer)
            else:
                layer = Conv2D(filters=number_of_filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding='same')(layer)
            layer = LeakyReLU(alpha=0.2)(layer)
            if normalization == True:
                layer = BatchNormalization(momentum=0.8)(layer)
            return(layer)

        image = Input(shape=self.high_resolution_image_size)

        model = build_layer(image, self.number_of_filters_at_base_layer[1],
          normalization = False)
        model = build_layer(model, self.number_of_filters_at_base_layer[1],
          strides=2)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 2)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 2,
          strides=2)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 4)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 4,
          strides=2)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 8)
        model = build_layer(model, self.number_of_filters_at_base_layer[1] * 8,
          strides=2)

        model = Dense(units=self.number_of_filters_at_base_layer[1] * 16)(model)
        model = LeakyReLU(alpha=0.2)(model)
        validity = Dense(units=1, activation = 'sigmoid')(model)

        discriminator = Model(inputs=image, outputs=validity)
        return(discriminator)

    def train(self, X_train_low_resolution, X_train_high_resolution, number_of_epochs, batch_size=128,
              sample_interval=None, sample_file_prefix='sample'):
        valid = np.ones((batch_size, *self.discriminator_patch_size))
        fake = np.zeros((batch_size, *self.discriminator_patch_size))

        for epoch in range(number_of_epochs):

            indices = np.random.randint(0, X_train_low_resolution.shape[0] - 1, batch_size)

            low_resolution_images = None
            high_resolution_images = None
            if self.dimensionality == 2:
                low_resolution_images = X_train_low_resolution[indices,:,:,:]
                high_resolution_images = X_train_high_resolution[indices,:,:,:]
            else:
                low_resolution_images = X_train_low_resolution[indices,:,:,:,:]
                high_resolution_images = X_train_high_resolution[indices,:,:,:,:]

            # train discriminator

            fake_high_resolution_images = self.generator.predict(low_resolution_images)

            d_loss_real = self.discriminator.train_on_batch(high_resolution_images, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_high_resolution_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator
            g_loss = None
            if self.use_image_net_weights == True:
                image_features = self.vgg_model.predict(high_resolution_images)
                g_loss = self.combined_model.train_on_batch(
                  [low_resolution_images, high_resolution_images], [valid, image_features])
                print("Epoch ", epoch, ": [Discriminator loss: ", d_loss[0],
                    "] ", "[Generator loss: ", g_loss[0], "] ")
            else:
                g_loss = self.combined_model.train_on_batch(
                  [low_resolution_images, high_resolution_images], valid)
                print("Epoch ", epoch, ": [Discriminator loss: ", d_loss[0],
                    "] ", "[Generator loss: ", g_loss, "] ")


            if self.dimensionality == 2:
                if sample_interval != None:
                    if epoch % sample_interval == 0:

                        # Do a 2x3 grid
                        #
                        # low res image | high res image | original high res image
                        # low res image | high res image | original high res image

                        X = list()

                        index = np.random.randint(0, X_train_low_resolution.shape[0] - 1, 1)
                        low_resolution_image = X_train_low_resolution[index,:,:,:]
                        high_resolution_image = X_train_high_resolution[index,:,:,:]

                        X.append(self.generator.predict(low_resolution_image))
                        X.append(high_resolution_image)

                        index = np.random.randint(0, X_train_low_resolution.shape[0] - 1, 1)
                        low_resolution_image = X_train_low_resolution[index,:,:,:]
                        high_resolution_image = X_train_high_resolution[index,:,:,:]

                        X.append(self.generator.predict(low_resolution_image))
                        X.append(high_resolution_image)

                        plot_images = np.concatenate(X)
                        plot_images = 0.5 * plot_images + 0.5

                        titles = ['Predicted', 'Original']
                        figure, axes = plot.subplots(2, 2)

                        count = 0
                        for i in range(2):
                            for j in range(2):
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

