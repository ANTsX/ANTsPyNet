import numpy as np
import tarfile

def ukbb(image,
         pipeline="All",
         target="Age",
         antsxnet_cache_directory=None,
         verbose=False):

    """
    Perform arterial lesion segmentation using U-net.

    Arguments
    ---------
    image : ANTsImage
        input image

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = arterial_lesion_segmentation(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if image.dimension != 2:
        raise ValueError( "Image dimension must be 2." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    channel_size = 1

    weights_file_name = get_pretrained_network("arterialLesionWeibinShi",
        antsxnet_cache_directory=antsxnet_cache_directory)

    resampled_image_size = (512, 512)

    unet_model = create_unet_model_2d((*resampled_image_size, channel_size),
        number_of_outputs=1, mode="sigmoid",
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        dropout_rate=0.0, weight_decay=0,
        additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    unet_model.load_weights(weights_file_name)

    if verbose == True:
        print("Preprocessing:  Resampling and N4 bias correction.")

    preprocessed_image = ants.image_clone(image)
    preprocessed_image = preprocessed_image / preprocessed_image.max()
    preprocessed_image = ants.resample_image(preprocessed_image, resampled_image_size, use_voxels=True, interp_type=0)
    mask = ants.image_clone(preprocessed_image) * 0 + 1
    preprocessed_image = ants.n4_bias_field_correction(preprocessed_image, mask=mask, shrink_factor=2, return_bias_field=False, verbose=verbose)

    batchX = np.expand_dims(preprocessed_image.numpy(), axis=0)
    batchX = np.expand_dims(batchX, axis=-1)
    batchX = (batchX - batchX.min()) / (batchX.max() - batchX.min())

    predicted_data = unet_model.predict(batchX, verbose=int(verbose))

    origin = preprocessed_image.origin
    spacing = preprocessed_image.spacing
    direction = preprocessed_image.direction

    foreground_probability_image = ants.from_numpy(np.squeeze(predicted_data[0, :, :, 0]),
        origin=origin, spacing=spacing, direction=direction)

    if verbose == True:
        print("Post-processing:  resampling to original space.")

    foreground_probability_image = ants.resample_image_to_target(foreground_probability_image, image)

    return(foreground_probability_image)


def e13x5_brain_extraction(image,
                           view = "sagittal",
                           which_axis=2,
                           antsxnet_cache_directory=None,
                           verbose=False):

    """
    Perform brain extraction of Allen's E13.5 mouse embroyonic data.

    Arguments
    ---------
    image : ANTsImage
        input image

    view : string
        Two trained networks are available:  "coronal" or "sagittal".

    which_axis : integer
        If 3-D image, which_axis specifies the direction of the "view".

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = allen_e13x5_brain_extraction(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = ""
    if view.lower() == "coronal":  
        weights_file_name = get_pretrained_network("e13x5_coronal_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    elif view.lower() == "sagittal":
        weights_file_name = get_pretrained_network("e13x5_sagittal_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    else:
        raise ValueError("Valid view options are coronal and sagittal.")

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=2, mode="classification",
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        dropout_rate=0.0, weight_decay=0,
        additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    unet_model.load_weights(weights_file_name)

    if verbose:
        print("Preprocessing:  Resampling.")

    number_of_channels = image.components
    number_of_slices = 1
    if image.dimension > 2:
        number_of_slices = image.shape[which_axis]
    
    image_channels = list()
    if number_of_channels == 1:
        image_channels.append(image)
    else:
        image_channels = ants.split_channels(image)
     
    batch_X = np.zeros((number_of_channels * number_of_slices, *resampled_image_size, 1))
    
    count = 0
    for i in range(number_of_channels):
        image_channel_array = image_channels[i].numpy()
        for j in range(number_of_slices):
            slice = None
            if image.dimension > 2:
                if which_axis == 0:
                    image_channel_slice_array = np.squeeze(image_channel_array[j,:,:])
                elif which_axis == 1:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,j,:])
                else:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,:,j])
                slice = ants.from_numpy(image_channel_slice_array)
            else:
                slice = image_channels[i]
            slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
            slice_array = slice_resampled.numpy()
            if slice_array.max() > slice_array.min():
                slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
            batch_X[count,:,:,0] = slice_array
            count = count + 1

    if verbose:
        print("Prediction: ")

    predicted_data = unet_model.predict(batch_X, verbose=int(verbose))
    if number_of_channels > 1:
        if verbose:
            print("Averaging across channels.")
        predicted_data_temp = np.split(predicted_data, number_of_channels, axis=0) 
        predicted_data = np.zeros((number_of_slices, *resampled_image_size, 1))
        for i in range(number_of_channels):
            predicted_data = (predicted_data * i + predicted_data_temp[i]) / (i + 1)

    if verbose:
        print("Post-processing:  resampling to original space.")

    foreground_probability_array = np.zeros(image.shape)
    for j in range(number_of_slices):
        slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,1]))
        slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
        if image.dimension == 2: 
            foreground_probability_array[:,:] = slice.numpy()
        else:
            if which_axis == 0:
                foreground_probability_array[j,:,:] = slice.numpy()
            elif which_axis == 1:
                foreground_probability_array[:,j,:] = slice.numpy()
            else:
                foreground_probability_array[:,:,j] = slice.numpy()

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    foreground_probability_image = ants.from_numpy(foreground_probability_array,
        origin=origin, spacing=spacing, direction=direction)

    return(foreground_probability_image)

