import numpy as np
import tensorflow as tf
import ants

def mri_super_resolution(image, 
                         expansion_factor=[1,1,2],
                         feature="vgg",
                         target_range=[1,0], 
                         poly_order='hist', 
                         verbose=False):

    """
    Perform super-resolution of MRI data using deep back projection 
    network.  Work described in
    
    https://www.medrxiv.org/content/10.1101/2023.02.02.23285376v1
    
    with the GitHub repo located at https://github.com/stnava/siq
    
    Note that some preprocessing possibilities for the input includes:
      * Truncate intensity (see ants.iMath(..., 'TruncateIntensity', ...)
    
    Arguments
    ---------
    image : ANTsImage
        magnetic resonance image
        
    expansion_factor : 3-tuple
        Specifies the increase in resolution per dimension.  Possibilities
        include:
          * [1,1,2]    
          * [1,1,3]    
          * [1,1,4]    
          * [1,1,6]    
          * [2,2,2]
          * [2,2,4]
          
    feature : string
        Type of network.  Choices include "grader" or "vgg".
        
    target_range : 2-tuple
        Range for apply_super_resolution_model.

    poly_order : int or 'hist'
        Parameter for regression matching or specification of histogram matching.
          
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

    from ..utilities import get_pretrained_network
    from ..utilities import apply_super_resolution_model_to_image

    if image.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    network_basename = ("sig_smallshort_train_" + 
                        'x'.join(map(str, expansion_factor)) + 
                        '_1chan_feat' + feature + 'L6_best_mdl')
    model_and_weights_filename = get_pretrained_network(network_basename)
    model_sr = tf.keras.models.load_model(model_and_weights_filename, compile=False)

    image_sr = apply_super_resolution_model_to_image(
        image, model_sr, target_range=target_range, regression_order=None, verbose=verbose)
    if poly_order is not None:
        if verbose:
            print("Match intensity with " + str(poly_order))
        if poly_order == "hist":
            if verbose:
                print("Histogram match input/output images.")
            image_sr = ants.histogram_match_image(image_sr, image)
        else:
            if verbose:
                print("Regression match input/output images.")
            image_resampled = ants.resample_image_to_target(image, image_sr)
            image_sr = ants.regression_match_image(image_sr, image_resampled, poly_order=poly_order)

    return image_sr
