
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.layers import Layer, InputSpec

class ResampleTensorLayer2D(Layer):

    """
    Tensor resampling layer (2D).

    Arguments
    ---------
    shape : tuple
        Specifies the output shape of the resampled tensor.

    interpolation_type : string
        One of 'nearest_neighbor', 'linear', or 'cubic'.

    Returns
    -------
    Keras layer
        A keras layer

    """

    def __init__(self, shape, interpolation_type='nearest_neighbor', name='', **kwargs):

        if len(shape) != 2:
            raise ValueError("shape must be of length 2 specifying the width and " +
                             "height of the resampled tensor.")
        self.shape = shape

        allowed_types = set(['nearest_neighbor', 'linear', 'cubic'])
        if not interpolation_type in allowed_types:
            raise ValueError("interpolation_type not one of the allowed types.")
        self.interpolation_type = interpolation_type

        self._name = name

        super(ResampleTensorLayer2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Input tensor must be of rank 4.")
        return((input_shape[0], self.shape[0], self.shape[1], input_shape[3]))

    def call(self, x, mask=None):
        dimensionality = 2

        new_size=self.shape
        input_shape = x.get_shape()
        old_size=input_shape[1:(dimensionality + 1)]

        if new_size == old_size:
            return(x + 0)

        resampled_tensor = None
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor = tf.image.resize(x, size=new_size, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor = tf.image.resize(x, size=new_size, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor = tf.image.resize(x, size=new_size, method='bicubic')

        return(resampled_tensor)

    def get_config(self):
        config = {"shape": self.shape, "interpolation_type": self.interpolation_type}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ResampleTensorLayer3D(Layer):

    """
    Tensor resampling layer (3D).

    Arguments
    ---------
    shape : tuple
        Specifies the output shape of the resampled tensor.

    interpolation_type : string
        One of 'nearest_neighbor', 'linear', or 'cubic'.

    Returns
    -------
    Keras layer
        A keras layer

    """

    def __init__(self, shape, interpolation_type='nearest_neighbor', name='', **kwargs):

        if len(shape) != 3:
            raise ValueError("shape must be of length 3 specifying the width, " +
                             "height, and depth of the resampled tensor.")
        self.shape = shape

        allowed_types = set(['nearest_neighbor', 'linear', 'cubic'])
        if not interpolation_type in allowed_types:
            raise ValueError("interpolation_type not one of the allowed types.")
        self.interpolation_type = interpolation_type

        self._name = name

        super(ResampleTensorLayer3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError("Input tensor must be of rank 5.")
        return((input_shape[0], self.shape[0], self.shape[1], self.shape[2],
                input_shape[4]))

    def call(self, x, mask=None):
        dimensionality = 3

        new_size=self.shape
        input_shape = x.get_shape()
        old_size = input_shape[1:(dimensionality + 1)]

        channel_size = input_shape[-1]

        if new_size == old_size:
            return(x + 0)

        resampled_tensor = None

        # Do yz
        squeeze_tensor_yz = tf.reshape(x, (-1, tf.shape(x)[2], tf.shape(x)[3], channel_size))

        new_shape_yz = (new_size[1], new_size[2])

        resampled_tensor_yz = None
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bicubic')

        new_shape_yz = (-1, tf.shape(x)[1], new_size[1], new_size[2], channel_size)
        resume_tensor_yz = tf.reshape(resampled_tensor_yz, new_shape_yz)

        # Do x

        reoriented_tensor = tf.transpose(resume_tensor_yz, (0, 3, 2, 1, 4))

        squeeze_tensor_x = tf.reshape(reoriented_tensor, (-1, new_size[1], tf.shape(x)[1], channel_size))

        new_shape_x = (new_size[1], new_size[0])

        resampled_tensor_x = None
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bicubic')

        new_shape_x = (-1, new_size[2], new_size[1], new_size[0], channel_size)
        resumeTensor_x = tf.reshape(resampled_tensor_x, new_shape_x)

        resampled_tensor = tf.transpose(resumeTensor_x, (0, 3, 2, 1, 4))

        return(resampled_tensor)

    def get_config(self):
        config = {"shape": self.shape, "interpolation_type": self.interpolation_type}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
