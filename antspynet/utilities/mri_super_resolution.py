import numpy as np
import tensorflow as tf
import ants

from warnings import warn

def mri_super_resolution(image, verbose=False):

    """
    Perform super-resolution (2x) of MRI data using deep back projection network.

    Arguments
    ---------
    image : ANTsImage
        magnetic resonance image

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    The super-resolved image.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> image_sr = mri_super_resolution(image)
    """

    warn('This method is deprecated.  Please see https://github.com/stnava/siq', 
         DeprecationWarning, stacklevel=2)

    from ..utilities import get_pretrained_network
    from ..utilities import apply_super_resolution_model_to_image
    from ..utilities import regression_match_image

    if image.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    model_and_weights_file_name = get_pretrained_network("mriSuperResolution")
    model_sr = tf.keras.models.load_model(model_and_weights_file_name, compile=False)

    image_sr = apply_super_resolution_model_to_image(
        image, model_sr, target_range=(-127.5, 127.5)
    )
    image_sr = regression_match_image(
        image_sr, ants.resample_image_to_target(image, image_sr), poly_order=1
    )

    return image_sr
