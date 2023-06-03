import numpy as np
import ants
import warnings

def arterial_lesion_segmentation(image,
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


def allen_ex5_brain_extraction(image,
                               view = "sagittal",
                               which_axis=2,
                               antsxnet_cache_directory=None,
                               verbose=False):

    """
    Perform brain extraction of Allen's E13.5 and E15.5 mouse embroyonic data.

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
        weights_file_name = get_pretrained_network("ex5_coronal_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    elif view.lower() == "sagittal":
        weights_file_name = get_pretrained_network("ex5_sagittal_weights",
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
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_array = slice_resampled.numpy()
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

def allen_histology_brain_mask(image,
                               which_axis=2,
                               antsxnet_cache_directory=None,
                               verbose=False):

    """
    Determine brain foreground of Allen's mouse data.

    Arguments
    ---------
    image : ANTsImage
        input image

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
    >>> output = allen_histology_brain_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = get_pretrained_network("allen_brain_mask_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

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
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_array = slice_resampled.numpy()
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

def allen_histology_hemispherical_coronal_mask(image,
                                               which_axis=2,
                                               antsxnet_cache_directory=None,
                                               verbose=False):

    """
    Determine left and right hemisphere brain masks of Allen's mouse data in coronal 
    acquisitions for both P* and E*x5 data.  This assumes that the original histology
    image has been pre-extracted.

    Arguments
    ---------
    image : ANTsImage
        input image

    which_axis : integer
        If 3-D image, which_axis specifies the direction of the coronal view.

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
    >>> output = allen_histology_hemispherical_coronal_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = get_pretrained_network("allen_brain_leftright_coronal_mask_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    classes = (0, 1, 2)
    number_of_classification_labels = len(classes)

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=number_of_classification_labels, mode="classification",
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
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_smoothed = ants.smooth_image(slice_resampled, 1.0)
                slice_array = slice_smoothed.numpy()
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

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    probability_images = list()
    for i in range(len(classes)):
        if verbose == True:
            print("Reconstructing image", classes[i])

        probability_image_array = np.zeros(image.shape)
        for j in range(number_of_slices):
            slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,i]))
            slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
            if image.dimension == 2: 
                probability_image_array[:,:] = slice.numpy()
            else:
                if which_axis == 0:
                    probability_image_array[j,:,:] = slice.numpy()
                elif which_axis == 1:
                    probability_image_array[:,j,:] = slice.numpy()
                else:
                    probability_image_array[:,:,j] = slice.numpy()

        probability_images.append(ants.from_numpy(probability_image_array,
            origin=origin, spacing=spacing, direction=direction))

    image_matrix = ants.image_list_to_matrix(probability_images, image_channels[0] * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), image_channels[0] * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return(return_dict)


def allen_histology_cerebellum_mask(image,
                                    which_axis=2,
                                    view = 'sagittal',
                                    antsxnet_cache_directory=None,
                                    verbose=False):

    """
    Determine cerebellum foreground of Allen's mouse data.

    Arguments
    ---------
    image : ANTsImage
        input image

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
    >>> output = allen_histology_cerebellum_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = None
    if view == "sagittal":
        weights_file_name = get_pretrained_network("allen_cerebellum_sagittal_mask_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    elif view == "coronal":   
        weights_file_name = get_pretrained_network("allen_cerebellum_coronal_mask_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    else:
        raise ValueError("Unrecognized option for view.  Must be sagittal or coronal.")    

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=1, mode="sigmoid",
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
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_array = slice_resampled.numpy()
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
        slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,0]))
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

def allen_histology_super_resolution(image,
                                     antsxnet_cache_directory=None,
                                     verbose=False):

    """
    Super resolution (2x) of a single image slice (256x256 -> 512x512)

    Arguments
    ---------
    image : ANTsImage or ANTsImage list
        input image or input image list

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Super resolution image of size 512x512 voxels (or a list depending on the input)

    Example
    -------
    >>> output = allen_histology_super_resolution(histology_image)
    """

    from ..architectures import create_deep_back_projection_network_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import regression_match_image

    image_list = list()
    if isinstance(image, list):
        image_list = image
    else:
        image_list.append(image)

    lr_image_size = (256, 256)
    sr_image_size = (512, 512)
    
    image_lr_list = list()
    for i in range(len(image_list)):

        if image_list[i].components != 3:
            raise ValueError("Number of image channels should be 3 (rgb).")
        if image_list[i].dimension != 2:
            raise ValueError("Input image should be 2-D.")
                
        do_resample = False
        if image_list[i].shape != lr_image_size:
            warnings.warn("Resampling input image to (256, 256).")
            do_resample = True

        image_lr = ants.image_clone(image_list[i])
        image_lr_channels = ants.split_channels(image_lr)
        for c in range(len(image_lr_channels)):
            if do_resample:
                image_lr_channels[c] = ants.resample_image(image_lr_channels[c], resample_params=lr_image_size, 
                                                           use_voxels=True, interp_type=0)
            image_lr_channels[c] = ((image_lr_channels[c] - image_lr_channels[c].min()) / 
                                    (image_lr_channels[c].max() - image_lr_channels[c].min()))
        image_lr = ants.merge_channels(image_lr_channels)
        image_lr_list.append(image_lr)

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    weights_file_name = get_pretrained_network("allen_sr_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

    sr_model = create_deep_back_projection_network_model_2d((*lr_image_size, 3),
        number_of_outputs=3, convolution_kernel_size=(6, 6), strides=(2, 2))
    sr_model.load_weights(weights_file_name)

    batch_X = np.zeros((len(image_lr_list), *lr_image_size, 3))
    for i in range(len(image_lr_list)):
        batch_X[i,:,:,:] = image_lr_list[i].numpy()
    
    if verbose:
        print("Prediction: ")

    predicted_data = sr_model.predict(batch_X, verbose=int(verbose))
    
    if verbose:
        print("Regression match output image.")

    spacing_factor = (lr_image_size[0] - 1) / (sr_image_size[0] - 1)
    
    image_sr_list = list()
    for i in range(len(image_lr_list)):
        image_sr = ants.from_numpy(predicted_data[i,:,:,:], origin=image_list[i].origin,
                                   direction=image_list[i].direction, 
                                   spacing=(spacing_factor * image_list[i].spacing[0], 
                                            spacing_factor * image_list[i].spacing[1]),
                                   has_components=True)
        image_channels = ants.split_channels(image_list[i])
        image_sr_channels = ants.split_channels(image_sr)
        for c in range(len(image_sr_channels)):
            image_lr_channel_resampled = ants.resample_image(image_channels[c], resample_params=sr_image_size, 
                                                             use_voxels=True, interp_type=0)
            image_sr_channels[c] = regression_match_image(image_sr_channels[c], image_lr_channel_resampled)
        image_sr_list.append(ants.merge_channels(image_sr_channels))

    if isinstance(image, list):
        return(image_sr_list)
    else:
        return(image_sr_list[0])
