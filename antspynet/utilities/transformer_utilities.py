import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense, Embedding,
                                     Layer, MaxPool2D, MaxPool3D,
                                     ZeroPadding2D, ZeroPadding3D)

##########
#
# Taken from:
#
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
#
# and
#
# https://keras.io/examples/vision/cct/
#
##########

class ExtractPatches2D(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class ExtractPatches3D(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.extract_volume_patches(
            input=images,
            ksizes=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, self.patch_size, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class EncodePatches(Layer):
    def __init__(self, number_of_patches, projection_dimension):
        super().__init__()
        self.number_of_patches = number_of_patches
        self.projection = Dense(units=projection_dimension)
        self.position_embedding = Embedding(
            input_dim=number_of_patches, output_dim=projection_dimension
        )
    def call(self, patch):
        positions = tf.range(start=0, limit=self.number_of_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class StochasticDepth(Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

class ExtractConvolutionalPatches2D(Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        number_of_filters=[64, 128],
        do_positional_embedding=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is our tokenizer.

        number_of_conv_layers = len(number_of_filters)

        self.conv_model = keras.Sequential()
        for i in range(number_of_conv_layers):
            self.conv_model.add(
                Conv2D(
                    number_of_filters[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(ZeroPadding2D(padding))
            self.conv_model.add(
                MaxPool2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.do_positional_embedding = do_positional_embedding

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]),
        )
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.do_positional_embedding:
            dummy_inputs = tf.ones((1, *image_size))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dimension = tf.shape(dummy_outputs)[-1]

            embedding_layer = Embedding(
                input_dim=sequence_length, output_dim=projection_dimension
            )
            return embedding_layer, sequence_length
        else:
            return None

class ExtractConvolutionalPatches3D(Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        number_of_filters=[64, 128],
        do_positional_embedding=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is our tokenizer.

        number_of_conv_layers = len(number_of_filters)

        self.conv_model = keras.Sequential()
        for i in range(number_of_conv_layers):
            self.conv_model.add(
                Conv3D(
                    number_of_filters[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(ZeroPadding3D(padding))
            self.conv_model.add(
                MaxPool3D(pooling_kernel_size, pooling_stride, "same")
            )

        self.do_positional_embedding = do_positional_embedding

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2] * tf.shape(outputs)[3], tf.shape(outputs)[-1]),
        )
        return reshaped

    def positional_embedding(self, image_size):
        # Positional embeddings are optional in CCT. Here, we calculate
        # the number of sequences and initialize an `Embedding` layer to
        # compute the positional embeddings later.
        if self.do_positional_embedding:
            dummy_inputs = tf.ones((1, *image_size))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dimension = tf.shape(dummy_outputs)[-1]

            embedding_layer = Embedding(
                input_dim=sequence_length, output_dim=projection_dimension
            )
            return embedding_layer, sequence_length
        else:
            return None

