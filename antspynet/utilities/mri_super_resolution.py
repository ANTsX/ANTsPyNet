import os

import numpy as np
import keras

import ants

def mri_super_resolution(image,
                         output_directory=None,
                         verbose=False):

    """
    Perform super-resolution (2x) of MRI data using deep back projection network.

    Arguments
    ---------
    image : ANTsImage
        magnetic resonance image

    output_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        tempfile.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> image_sr = mri_super_resolution(image)
    """

    from ..utilities import get_pretrained_network
    from ..utilities import apply_super_resolution_model_to_image
    from ..utilities import regression_match_image

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    model_and_weights_file_name = None
    if output_directory is not None:
        model_and_weights_file_name = output_directory + "/mindmapsSR_16_ANINN222_0.h5"
        if not os.path.exists(model_and_weights_file_name):
            if verbose == True:
                print("MRI super-resolution:  downloading model weights.")
            model_and_weights_file_name = get_pretrained_network("mriSuperResolution", model_and_weights_file_name)
    else:
        model_and_weights_file_name = get_pretrained_network("mriSuperResolution")

    model_sr = keras.models.load_model(model_and_weights_file_name, compile=False)

    image_sr = apply_super_resolution_model_to_image(image, model_sr, target_range=(-127.5, 127.5))
    image_sr = regression_match_image(image_sr, ants.resample_image_to_target(image, image_sr), poly_order=1)

    return(image_sr)