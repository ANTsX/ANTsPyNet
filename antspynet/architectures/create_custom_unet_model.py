from keras.models import Model
from keras.layers import (Input, Concatenate, ReLU,
                          Conv3D, Conv3DTranspose, MaxPooling3D)

def create_nobrainer_unet_model_3d(input_image_size,
                                   number_of_outputs=1,
                                   number_of_filters_at_base_layer=16,
                                   convolution_kernel_size=(3, 3, 3),
                                   deconvolution_kernel_size=(2, 2, 2),
                                  ):
    """
    Implementation of the "NoBrainer" U-net architecture

    Creates a keras model of the U-net deep learning architecture for image
    segmentation available at:

            https://github.com/neuronets/nobrainer/


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    number_of_outputs : integer
        Number of segmentation labels.

    number_of_filters_at_base_layer : integer
        number of filters at the beginning and end of the `U`.  Doubles at each
        descending/ascending layer.  Default = 16.

    convolution_kernel_size : tuple of length 3
        Defines the kernel size during the encoding.  Default = (3, 3, 3).

    deconvolution_kernel_size : tuple of length 3
        Defines the kernel size during the decoding.  Default = (2, 2, 2).

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> import requests
    >>> import tempfile
    >>> from os import path
    >>> import ants
    >>> import numpy as np
    >>>
    >>> model = create_nobrainer_unet_model_3d((None, None, None, 1))
    >>> model.summary()
    >>>
    >>> url_weights = "https://github.com/neuronets/nobrainer-models/releases/download/0.1/brain-extraction-unet-128iso-weights.h5"
    >>> temp_directory = tempfile.TemporaryDirectory()
    >>> target_file_weights = tempfile.NamedTemporaryFile(suffix=".h5", dir=temp_directory.name)
    >>> target_file_weights.close()
    >>> target_file_weights_name = target_file_weights.name
    >>> if not path.exists(target_file_weights_name):
    >>>     r = requests.get(url_weights)
    >>>     with open(target_file_weights_name, 'wb') as f:
    >>>         f.write(r.content)
    >>>
    >>> model.load_weights(target_file_weights_name)
    >>>
    >>> url_image = "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Example/1097782_defaced_MPRAGE.nii.gz?raw=true"
    >>> target_file_image = tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=temp_directory.name)
    >>> target_file_image.close()
    >>> target_file_image_name = target_file_image.name
    >>> if not path.exists(target_file_image_name):
    >>>     r = requests.get(url_image)
    >>>     with open(target_file_image_name, 'wb') as f:
    >>>         f.write(r.content)
    >>>
    >>> image = ants.image_read(target_file_image_name)
    >>> image = ants.image_math(image, 'Normalize') * 255.0
    >>> image_resampled = ants.resample_image(image, (256, 256, 256), True)
    >>> batchX = np.expand_dims(image_resampled.numpy(), axis=0)
    >>> batchX = np.expand_dims(batchX, axis=-1)
    >>>
    >>> brain_mask_array = model.predict(batchX, verbose=0)
    >>> brain_mask_resampled = ants.from_numpy(np.squeeze(brain_mask_array[0,:,:,:,0]),
          origin=image_resampled.origin, spacing=image_resampled.spacing,
          direction=image_resampled.direction)
    >>> brain_mask_image = ants.resample_image(brain_mask_resampled, image.shape, True, 1)
    >>> minimum_brain_volume = round( 649933.7 )
    >>> brain_mask_labeled = ants.label_clusters(brain_mask_image, minimum_brain_volume)
    >>> ants.image_write(brain_mask_labeled, "brain_mask.nii.gz")
    """

    inputs = Input(shape = input_image_size)

    # Encoding path

    outputs = Conv3D(filters=number_of_filters_at_base_layer,
                     kernel_size=convolution_kernel_size,
                     padding='same')(inputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip1 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip2 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    skip3 = outputs
    outputs = MaxPooling3D(pool_size=(2, 2, 2))(outputs)

    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*16,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    # Decoding path

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*16,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip3, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*8,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*8,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip2, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*4,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    outputs = Conv3DTranspose(filters=number_of_filters_at_base_layer*4,
                     kernel_size=deconvolution_kernel_size,
                     strides=2,
                     padding='same')(outputs)

    outputs = Concatenate()([skip1, outputs])
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)
    outputs = Conv3D(filters=number_of_filters_at_base_layer*2,
                     kernel_size=convolution_kernel_size,
                     padding='same')(outputs)
    outputs = ReLU()(outputs)

    convActivation = ''
    if number_of_outputs == 1:
        convActivation = 'sigmoid'
    else:
        convActivation = 'softmax'

    outputs = Conv3D(filters=number_of_outputs,
                     kernel_size=1,
                     activation = convActivation)(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return unet_model


