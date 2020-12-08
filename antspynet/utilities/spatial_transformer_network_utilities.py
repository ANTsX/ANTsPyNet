
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers

import tensorflow as tf

class SpatialTransformer2D(Layer):

    """
    Custom layer for the spatial transfomer network.

    Arguments
    ---------
    inputs : list of size 2
        The first element are the images and the second element are the
        weights.

    resampled_size : tuple of length 2
        Size of the resampled output images.

    transform_type : string
        Transform type (default = 'affine').

    interpolator_type : string
        Interpolator type (default = 'linear').

    Returns
    -------
    Keras layer
        A 2-D keras layer

    """

    def __init__(self, resampled_size, transform_type='affine', interpolator_type='linear', **kwargs):
        if K.backend() != 'tensorflow':
            raise ValueError("Tensorflow is required for this STN implementation.")
        if len(resampled_size) != 2:
            raise ValueError("Resampled size must be a vector of length 2 (for 2-D).")

        self.resampled_size = resampled_size
        self.transform_type = transform_type
        self.interpolator_type = interpolator_type

        super(SpatialTransformer2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpatialTransformer2D, self).build(input_shape)

    def call(self, inputs, mask=None):
        images = inputs[0]
        transform_parameters = inputs[1]

        output = None
        if self.transform_type == 'affine':
            output = self.affine_transform_images(images, transform_parameters, self.resampled_size)
        else:
            raise ValueError("Unsupported transform type.")

        input_shape = [K.shape(images)]
        return(tf.reshape(output, self.compute_output_shape(input_shape)))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], self.resampled_size[0], self.resampled_size[1], input_shape[0][-1])
        return(output_shape)

    def affine_transform_images(self, images, affine_transform_parameters, resampled_size):
        batch_size = K.int_shape(images)[0]
        number_of_channels = K.int_shape(images)[-1]
        if batch_size is None:
            transform_parameters = K.reshape(affine_transform_parameters, shape=(-1, 2, 3))
        else:
            transform_parameters = K.reshape(affine_transform_parameters, shape=(batch_size, 2, 3))

        regular_grid = self.make_regular_grid(resampled_size)
        sampled_grids = K.dot(transform_parameters, regular_grid)

        if self.interpolator_type == 'linear':
            interpolated_image = self.linear_interpolate(images, sampled_grids, resampled_size)
        else:
            raise ValueError("Unsupported interpolator type.")

        if batch_size is None:
            new_output_shape = (-1, resampled_size[0], resampled_size[1], number_of_channels)
        else:
            new_output_shape = (batch_size, resampled_size[0], resampled_size[1], number_of_channels)

        interpolated_image = K.reshape(interpolated_image, shape = new_output_shape)

        return(interpolated_image)

    def make_regular_grid(self, resampled_size):
        x_linear_space = tf.linspace(-1.0, 1.0, resampled_size[1])
        y_linear_space = tf.linspace(-1.0, 1.0, resampled_size[0])

        x_coords, y_coords = tf.meshgrid(x_linear_space, y_linear_space)
        x_coords = K.flatten(x_coords)
        y_coords = K.flatten(y_coords)

        ones = K.ones_like(x_coords)
        regular_grid = K.concatenate([x_coords, y_coords, ones], axis = 0)
        regular_grid = K.flatten(regular_grid)
        regular_grid = K.reshape(regular_grid,
          (3, resampled_size[0] * resampled_size[1]))

        return(regular_grid)

    def linear_interpolate(self, images, sampled_grids, resampled_size):
        batch_size = K.shape(images)[0]
        height = K.shape(images)[1]
        width = K.shape(images)[2]
        number_of_channels = K.shape(images)[3]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = 0.5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = 0.5 * (y + 1.0) * K.cast(height, dtype='float32')

        x0 = K.cast(x, dtype='int32')
        x1 = x0 + 1
        y0 = K.cast(y, dtype='int32')
        y1 = y0 + 1

        xMax = int(K.int_shape(images)[2] - 1)
        yMax = int(K.int_shape(images)[1] - 1)

        x0 = K.clip(x0, 0, xMax)
        x1 = K.clip(x1, 0, xMax)
        y0 = K.clip(y0, 0, yMax)
        y1 = K.clip(y1, 0, yMax)

        batch_pixels = K.arange(0, batch_size) * (height * width)
        batch_pixels = K.expand_dims(batch_pixels, axis = -1)
        base = K.repeat_elements(batch_pixels,
                                 rep=int(resampled_size[0] * resampled_size[1]), axis=1)
        base = K.flatten(base)

        indices00 = base + y0 * width + x0
        indices01 = base + y1 * width + x0
        indices10 = base + y0 * width + x1
        indices11 = base + y1 * width + x1

        flat_images = K.reshape(images, shape=(-1, number_of_channels))
        flat_images = K.cast(flat_images, dtype='float32')

        pixelValues00 = K.gather(flat_images, indices00)
        pixelValues01 = K.gather(flat_images, indices01)
        pixelValues10 = K.gather(flat_images, indices10)
        pixelValues11 = K.gather(flat_images, indices11)

        x0 = K.cast(x0, dtype='float32')
        x1 = K.cast(x1, dtype='float32')
        y0 = K.cast(y0, dtype='float32')
        y1 = K.cast(y1, dtype='float32')

        weight00 = K.expand_dims(((x1 - x) * (y1 - y)), axis=1)
        weight01 = K.expand_dims(((x1 - x) * (y - y0)), axis=1)
        weight10 = K.expand_dims(((x - x0) * (y1 - y)), axis=1)
        weight11 = K.expand_dims(((x - x0) * (y - y0)), axis=1)

        interpolatedValues00 = weight00 * pixelValues00
        interpolatedValues01 = weight01 * pixelValues01
        interpolatedValues10 = weight10 * pixelValues10
        interpolatedValues11 = weight11 * pixelValues11

        interpolatedValues = (interpolatedValues00 + interpolatedValues01 +
          interpolatedValues10 + interpolatedValues11)

        return(interpolatedValues)

class SpatialTransformer3D(Layer):

    """
    Custom layer for the spatial transfomer network.

    Arguments
    ---------
    inputs : list of size 2
        The first element are the images and the second element are the
        weights.

    resampled_size : tuple of length 3
        Size of the resampled output images.

    transform_type : string
        Transform type (default = 'affine').

    interpolator_type : string
        Interpolator type (default = 'linear').

    Returns
    -------
    Keras layer
        A 3-D keras layer

    """

    def __init__(self, resampled_size, transform_type='affine', interpolator_type='linear', **kwargs):
        if K.backend() != 'tensorflow':
            raise ValueError("Tensorflow is required for this STN implementation.")
        if len(resampled_size) != 3:
            raise ValueError("Resampled size must be a vector of length 3 (for 3-D).")

        self.resampled_size = resampled_size
        self.transform_type = transform_type
        self.interpolator_type = interpolator_type

        super(SpatialTransformer3D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpatialTransformer3D, self).build(input_shape)

    def call(self, inputs, mask=None):
        images = inputs[0]
        transform_parameters = inputs[1]

        output = None
        if self.transform_type == 'affine':
            output = self.affine_transform_images(images, transform_parameters, self.resampled_size)
        else:
            raise ValueError("Unsupported transform type.")

        input_shape = [K.shape(images)]
        return(tf.reshape(output, self.compute_output_shape(input_shape)))

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0], self.resampled_size[0],
                        self.resampled_size[1], self.resampled_size[2],
                        input_shape[0][-1])
        return(output_shape)

    def affine_transform_images(self, images, affine_transform_parameters, resampled_size):
        batch_size = K.int_shape(images)[0]
        number_of_channels = K.int_shape(images)[-1]
        if batch_size is None:
            transform_parameters = K.reshape(affine_transform_parameters, shape=(-1, 3, 4))
        else:
            transform_parameters = K.reshape(affine_transform_parameters, shape=(batch_size, 3, 4))

        regular_grid = self.make_regular_grid(resampled_size)
        sampled_grids = K.dot(transform_parameters, regular_grid)

        if self.interpolator_type == 'linear':
            interpolated_image = self.linear_interpolate(images, sampled_grids, resampled_size)
        else:
            raise ValueError("Unsupported interpolator type.")

        if batch_size is None:
            new_output_shape = (-1, resampled_size[0], resampled_size[1], resampled_size[2], number_of_channels)
        else:
            new_output_shape = (batch_size, resampled_size[0], resampled_size[1], resampled_size[2], number_of_channels)

        interpolated_image = K.reshape(interpolated_image, shape = new_output_shape)

        return(interpolated_image)

    def make_regular_grid(self, resampled_size):
        x_linear_space = tf.linspace(-1.0, 1.0, resampled_size[1])
        y_linear_space = tf.linspace(-1.0, 1.0, resampled_size[0])
        z_linear_space = tf.linspace(-1.0, 1.0, resampled_size[2])

        x_coords, y_coords, z_coords = tf.meshgrid(x_linear_space, y_linear_space, z_linear_space)
        x_coords = K.flatten(x_coords)
        y_coords = K.flatten(y_coords)
        z_coords = K.flatten(z_coords)

        ones = K.ones_like(x_coords)
        regular_grid = K.concatenate([x_coords, y_coords, z_coords, ones], axis = 0)
        regular_grid = K.flatten(regular_grid)
        regular_grid = K.reshape(regular_grid,
          (4, resampled_size[0] * resampled_size[1] * resampled_size[2]))

        return(regular_grid)

    def linear_interpolate(self, images, sampled_grids, resampled_size):
        batch_size = K.shape(images)[0]
        height = K.shape(images)[1]
        width = K.shape(images)[2]
        depth = K.shape(images)[3]
        number_of_channels = K.shape(images)[4]

        x = K.cast(K.flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = K.cast(K.flatten(sampled_grids[:, 1:2, :]), dtype='float32')
        z = K.cast(K.flatten(sampled_grids[:, 2:3, :]), dtype='float32')

        x = 0.5 * (x + 1.0) * K.cast(width, dtype='float32')
        y = 0.5 * (y + 1.0) * K.cast(height, dtype='float32')
        z = 0.5 * (z + 1.0) * K.cast(depth, dtype='float32')

        x0 = K.cast(x, dtype='int32')
        x1 = x0 + 1
        y0 = K.cast(y, dtype='int32')
        y1 = y0 + 1
        z0 = K.cast(z, dtype='int32')
        z1 = z0 + 1

        xMax = int(K.int_shape(images)[2] - 1)
        yMax = int(K.int_shape(images)[1] - 1)
        zMax = int(K.int_shape(images)[3] - 1)

        x0 = K.clip(x0, 0, xMax)
        x1 = K.clip(x1, 0, xMax)
        y0 = K.clip(y0, 0, yMax)
        y1 = K.clip(y1, 0, yMax)
        z0 = K.clip(z0, 0, zMax)
        z1 = K.clip(z1, 0, zMax)

        batch_pixels = K.arange(0, batch_size) * (height * width * depth)
        batch_pixels = K.expand_dims(batch_pixels, axis=-1)
        base = K.repeat_elements(batch_pixels,
                                 rep=int(resampled_size[0] * resampled_size[1] * resampled_size[2]), axis=1)
        base = K.flatten(base)

        indices000 = base + z0 * (width * height) + y0 * width + x0
        indices001 = base + z1 * (width * height) + y0 * width + x0
        indices010 = base + z0 * (width * height) + y1 * width + x0
        indices011 = base + z1 * (width * height) + y1 * width + x0
        indices100 = base + z0 * (width * height) + y0 * width + x1
        indices101 = base + z1 * (width * height) + y0 * width + x1
        indices110 = base + z0 * (width * height) + y1 * width + x1
        indices111 = base + z1 * (width * height) + y1 * width + x1

        flatImages = K.reshape(images, shape=(-1, number_of_channels))
        flatImages = K.cast(flatImages, dtype='float32')

        pixelValues000 = K.gather(flatImages, indices000)
        pixelValues001 = K.gather(flatImages, indices001)
        pixelValues010 = K.gather(flatImages, indices010)
        pixelValues011 = K.gather(flatImages, indices011)
        pixelValues100 = K.gather(flatImages, indices100)
        pixelValues101 = K.gather(flatImages, indices101)
        pixelValues110 = K.gather(flatImages, indices110)
        pixelValues111 = K.gather(flatImages, indices111)

        x0 = K.cast(x0, dtype='float32')
        x1 = K.cast(x1, dtype='float32')
        y0 = K.cast(y0, dtype='float32')
        y1 = K.cast(y1, dtype='float32')
        z0 = K.cast(z0, dtype='float32')
        z1 = K.cast(z1, dtype='float32')

        weight000 = K.expand_dims(((x1 - x) * (y1 - y) * (z1 - z)), axis = 1)
        weight001 = K.expand_dims(((x1 - x) * (y1 - y) * (z - z0)), axis = 1)
        weight010 = K.expand_dims(((x1 - x) * (y - y0) * (z1 - z)), axis = 1)
        weight011 = K.expand_dims(((x1 - x) * (y - y0) * (z - z0)), axis = 1)
        weight100 = K.expand_dims(((x - x0) * (y1 - y) * (z1 - z)), axis = 1)
        weight101 = K.expand_dims(((x - x0) * (y1 - y) * (z - z0)), axis = 1)
        weight110 = K.expand_dims(((x - x0) * (y - y0) * (z1 - z)), axis = 1)
        weight111 = K.expand_dims(((x - x0) * (y - y0) * (z - z0)), axis = 1)

        interpolatedValues000 = weight000 * pixelValues000
        interpolatedValues001 = weight001 * pixelValues001
        interpolatedValues010 = weight010 * pixelValues010
        interpolatedValues011 = weight011 * pixelValues011
        interpolatedValues100 = weight100 * pixelValues100
        interpolatedValues101 = weight101 * pixelValues101
        interpolatedValues110 = weight110 * pixelValues110
        interpolatedValues111 = weight111 * pixelValues111

        interpolatedValues = (
          interpolatedValues000 +
          interpolatedValues001 +
          interpolatedValues010 +
          interpolatedValues011 +
          interpolatedValues100 +
          interpolatedValues101 +
          interpolatedValues110 +
          interpolatedValues111)

        return(interpolatedValues)










