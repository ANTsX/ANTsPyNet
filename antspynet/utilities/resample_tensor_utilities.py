
import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.layers import Layer, InputSpec

#################################################################
#
#  Resampling layers (to a fixed size)
#
#################################################################

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
        resampled_tensor = None
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor = tf.image.resize(x, size=self.shape, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor = tf.image.resize(x, size=self.shape, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor = tf.image.resize(x, size=self.shape, method='bicubic')
        return(resampled_tensor)

    def get_config(self):
        config = {"shape": self.shape, "interpolation_type": self.interpolation_type}
        return dict(list(config.items()))

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
        return((input_shape[0], self.shape[0], self.shape[1], self.shape[2], input_shape[4]))

    def call(self, x, mask=None):

        channel_size = x.get_shape()[-1]

        resampled_tensor = None

        # Do yz

        new_shape_squeeze_yz = (-1, tf.shape(x)[2], tf.shape(x)[3], channel_size)
        squeeze_tensor_yz = tf.reshape(x, new_shape_squeeze_yz)

        resampled_tensor_yz = None
        new_shape_yz = (self.shape[1], self.shape[2])
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bicubic')

        new_shape_yz = (-1, tf.shape(x)[1], self.shape[1], self.shape[2], channel_size)
        resume_tensor_yz = tf.reshape(resampled_tensor_yz, new_shape_yz)

        # Do x

        reoriented_tensor = tf.transpose(resume_tensor_yz, (0, 3, 2, 1, 4))

        new_shape_squeeze_x = (-1, self.shape[1], tf.shape(x)[1], channel_size)
        squeeze_tensor_x = tf.reshape(reoriented_tensor, new_shape_squeeze_x)

        resampled_tensor_x = None
        new_shape_x = (self.shape[1], self.shape[0])
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bicubic')

        new_shape_x = (-1, self.shape[2], self.shape[1], self.shape[0], channel_size)
        resumeTensor_x = tf.reshape(resampled_tensor_x, new_shape_x)

        resampled_tensor = tf.transpose(resumeTensor_x, (0, 3, 2, 1, 4))

        return(resampled_tensor)

    def get_config(self):
        config = {"shape": self.shape, "interpolation_type": self.interpolation_type}
        return dict(list(config.items()))

#################################################################
#
#  Resampling layers (to a target tensor)
#
#################################################################

class ResampleTensorToTargetTensorLayer2D(Layer):

    """
    Tensor resampling layer (2D).

    Arguments
    ---------

    interpolation_type : string
        One of 'nearest_neighbor', 'linear', or 'cubic'.

    Returns
    -------
    Keras layer
        A keras layer

    """

    def __init__(self, interpolation_type='nearest_neighbor', name='', **kwargs):

        allowed_types = set(['nearest_neighbor', 'linear', 'cubic'])
        if not interpolation_type in allowed_types:
            raise ValueError("interpolation_type not one of the allowed types.")
        self.interpolation_type = interpolation_type

        self._name = name
        self.resampled_tensor = None

        super(ResampleTensorToTargetTensorLayer2D, self).__init__(**kwargs)

    def call(self, x, mask=None):

        source_tensor = x[0]
        target_tensor = x[1]

        new_shape = (tf.shape(target_tensor)[1], tf.shape(target_tensor)[2])

        if self.interpolation_type == 'nearest_neighbor':
            self.resampled_tensor = tf.image.resize(source_tensor, size=new_shape, method='nearest')
        elif self.interpolation_type == 'linear':
            self.resampled_tensor = tf.image.resize(source_tensor, size=new_shape, method='bilinear')
        elif self.interpolation_type == 'cubic':
            self.resampled_tensor = tf.image.resize(source_tensor, size=new_shape, method='bicubic')
        return(self.resampled_tensor)

    def compute_output_shape(self, input_shape):
        return(K.int_shape(self.resampled_tensor))

    def get_config(self):
        config = {"interpolation_type": self.interpolation_type}
        return dict(list(config.items()))

class ResampleTensorToTargetTensorLayer3D(Layer):

    """
    Tensor resampling layer (3D).

    Arguments
    ---------

    interpolation_type : string
        One of 'nearest_neighbor', 'linear', or 'cubic'.

    Returns
    -------
    Keras layer
        A keras layer

    """

    def __init__(self, interpolation_type='nearest_neighbor', name='', **kwargs):

        allowed_types = set(['nearest_neighbor', 'linear', 'cubic'])
        if not interpolation_type in allowed_types:
            raise ValueError("interpolation_type not one of the allowed types.")
        self.interpolation_type = interpolation_type

        self._name = name
        self.resampled_tensor = None

        super(ResampleTensorToTargetTensorLayer3D, self).__init__(**kwargs)

    def call(self, x, mask=None):

        source_tensor = x[0]
        target_tensor = x[1]

        channel_size = source_tensor.get_shape()[-1]

        # Do yz

        new_shape_squeeze_yz = (-1, tf.shape(source_tensor)[2], tf.shape(source_tensor)[3], channel_size)
        squeeze_tensor_yz = tf.reshape(source_tensor, new_shape_squeeze_yz)

        resampled_tensor_yz = None
        new_shape_yz = (tf.shape(target_tensor)[2], tf.shape(target_tensor)[3])
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_yz = tf.image.resize(squeeze_tensor_yz, size=new_shape_yz, method='bicubic')

        new_shape_yz = (-1, tf.shape(source_tensor)[1], tf.shape(target_tensor)[2], tf.shape(target_tensor)[3], channel_size)
        resume_tensor_yz = tf.reshape(resampled_tensor_yz, new_shape_yz)

        # Do x

        reoriented_tensor = tf.transpose(resume_tensor_yz, (0, 3, 2, 1, 4))

        new_shape_squeeze_x = (-1, tf.shape(target_tensor)[2], tf.shape(source_tensor)[1], channel_size)
        squeeze_tensor_x = tf.reshape(reoriented_tensor, new_shape_squeeze_x)

        resampled_tensor_x = None
        new_shape_x = (tf.shape(target_tensor)[2], tf.shape(target_tensor)[1])
        if self.interpolation_type == 'nearest_neighbor':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='nearest')
        elif self.interpolation_type == 'linear':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bilinear')
        elif self.interpolation_type == 'cubic':
            resampled_tensor_x = tf.image.resize(squeeze_tensor_x, size=new_shape_x, method='bicubic')

        new_shape_x = (-1, tf.shape(target_tensor)[3], tf.shape(target_tensor)[2], tf.shape(target_tensor)[1], channel_size)
        resumeTensor_x = tf.reshape(resampled_tensor_x, new_shape_x)

        self.resampled_tensor = tf.transpose(resumeTensor_x, (0, 3, 2, 1, 4))

        return(self.resampled_tensor)

    def compute_output_shape(self, input_shape):
        return(K.int_shape(self.resampled_tensor))

    def get_config(self):
        config = {"interpolation_type": self.interpolation_type}
        return dict(list(config.items()))

