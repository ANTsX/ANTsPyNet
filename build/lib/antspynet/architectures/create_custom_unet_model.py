from keras.models import Model
from keras.layers import (Add, Activation, Concatenate, ReLU, LeakyReLU,
                          Conv3D, Conv3DTranspose, Input, MaxPooling3D,
                          SpatialDropout3D, UpSampling3D)
from ..utilities import InstanceNormalization

def create_nobrainer_unet_model_3d(input_image_size):
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

    number_of_outputs = 1
    number_of_filters_at_base_layer = 16
    convolution_kernel_size = (3, 3, 3)
    deconvolution_kernel_size = (2, 2, 2)

    inputs = Input(shape=input_image_size)

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

def create_hippmapp3r_unet_model_3d(input_image_size,
                                    do_first_network=True,
                                    data_format="channels_last"):
    """
    Implementation of the "HippMapp3r" U-net architecture

    Creates a keras model implementation of the u-net architecture
    described here:

        https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.24811

    with the implementation available here:

        https://github.com/mgoubran/HippMapp3r

    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).

    do_first_network : boolean
        Boolean dictating if the model built should be the first (initial) network
        or second (refinement) network.

    data_format : string
        One of "channels_first" or "channels_last".  We do this for this specific
        architecture as the original weights were saved in "channels_first" format.

    Returns
    -------
    Keras model
        A 3-D keras model defining the U-net network.

    Example
    -------
    >>> shape_initial_stage = (160, 160, 128)
    >>> model_refine_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), False)
    >>> model_refine_stage.load_weights(get_pretrained_network("hippMapp3rRefine"))
    >>> shape_refine_stage = (112, 112, 64)
    >>> model_refine_stage = antspynet.create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), False)
    >>> model_refine_stage.load_weights(get_pretrained_network("hippMapp3rRefine"))
    """

    channels_axis = None
    if data_format == "channels_last":
        channels_axis = 4
    elif data_format == "channels_first":
        channels_axis = 1
    else:
        raise ValueError("Unexpected string for data_format.")

    def convB_3d_layer(input, number_of_filters, kernel_size=3, strides=1):
        block = Conv3D(filters=number_of_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       data_format=data_format)(input)
        block = InstanceNormalization(axis=channels_axis)(block)
        block = LeakyReLU()(block)
        return(block)

    def residual_block_3d(input, number_of_filters):
        block = convB_3d_layer(input, number_of_filters)
        block = SpatialDropout3D(rate=0.3,
                                 data_format=data_format)(block)
        block = convB_3d_layer(block, number_of_filters)
        return(block)

    def upsample_block_3d(input, number_of_filters):
        block = UpSampling3D(data_format=data_format)(input)
        block = convB_3d_layer(block, number_of_filters)
        return(block)

    def feature_block_3d(input, number_of_filters):
        block = convB_3d_layer(input, number_of_filters)
        block = convB_3d_layer(block, number_of_filters, kernel_size=1)
        return(block)

    number_of_filters_at_base_layer = 16

    number_of_layers = 6
    if do_first_network == False:
        number_of_layers = 5

    inputs = Input(shape=input_image_size)

    # Encoding path

    add = None

    encoding_convolution_layers = []
    for i in range(number_of_layers):
        number_of_filters = number_of_filters_at_base_layer * 2 ** i
        conv = None
        if i == 0:
            conv = convB_3d_layer(inputs, number_of_filters)
        else:
            conv = convB_3d_layer(add, number_of_filters, strides=2)
        residual_block = residual_block_3d(conv, number_of_filters)
        add = Add()([conv, residual_block])
        encoding_convolution_layers.append(add)

    # Decoding path

    outputs = encoding_convolution_layers[number_of_layers-1]

    # 256
    number_of_filters = (number_of_filters_at_base_layer *
      2 ** (number_of_layers - 2))
    outputs = upsample_block_3d(outputs, number_of_filters)

    if do_first_network == True:
        # 256, 128
        outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[4], outputs])
        outputs = feature_block_3d(outputs, number_of_filters)
        number_of_filters = int(number_of_filters / 2)
        outputs = upsample_block_3d(outputs, number_of_filters)

    # 128, 64
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[3], outputs])
    outputs = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(outputs, number_of_filters)

    # 64, 32
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[2], outputs])
    feature64 = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(feature64, number_of_filters)
    back64 = None
    if do_first_network == True:
        back64 = convB_3d_layer(feature64, 1, 1)
    else:
        back64 = Conv3D(filters=1,
                        kernel_size=1,
                        data_format=data_format)(feature64)
    back64 = UpSampling3D(data_format=data_format)(back64)

    # 32, 16
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[1], outputs])
    feature32 = feature_block_3d(outputs, number_of_filters)
    number_of_filters = int(number_of_filters / 2)
    outputs = upsample_block_3d(feature32, number_of_filters)
    back32 = None
    if do_first_network == True:
        back32 = convB_3d_layer(feature32, 1, 1)
    else:
        back32 = Conv3D(filters=1,
                        kernel_size=1,
                        data_format=data_format)(feature32)
    back32 = Add()([back64, back32])
    back32 = UpSampling3D(data_format=data_format)(back32)

    # Final
    outputs = Concatenate(axis=channels_axis)([encoding_convolution_layers[0], outputs])
    outputs = convB_3d_layer(outputs, number_of_filters, 3)
    outputs = convB_3d_layer(outputs, number_of_filters, 1)
    if do_first_network == True:
        outputs = convB_3d_layer(outputs, 1, 1)
    else:
        outputs = Conv3D(filters=1,
                         kernel_size=1,
                         data_format=data_format)(outputs)
    outputs = Add()([back32, outputs])
    outputs = Activation('sigmoid')(outputs)

    unet_model = Model(inputs=inputs, outputs=outputs)

    return(unet_model)








