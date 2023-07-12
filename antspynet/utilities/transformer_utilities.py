import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Embedding

##########
#
# Taken from:
#
# https://keras.io/examples/vision/image_classification_with_vision_transformer/
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