import numpy as np
import ants

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

