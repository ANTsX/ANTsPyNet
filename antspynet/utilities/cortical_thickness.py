import numpy as np
import tensorflow as tf
import ants

def cortical_thickness(t1,
                       antsxnet_cache_directory=None,
                       verbose=False):

    """
    Perform KellyKapowski cortical thickness using deep_atropos for
    segmentation.  Description concerning implementaiton and evaluation:

    https://www.medrxiv.org/content/10.1101/2020.10.19.20215392v1

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D unprocessed T1-weighted brain image.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Cortical thickness image and segmentation probability images.

    Example
    -------
    >>> image = ants.image_read( "t1w_image.nii.gz" )
    >>> kk = cortical_thickness( image )
    """

    from ..utilities import deep_atropos

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    atropos = deep_atropos(t1, do_preprocessing=True,
        antsxnet_cache_directory=antsxnet_cache_directory, verbose=True)

    # Kelly Kapowski cortical thickness

    kk_segmentation = atropos['segmentation_image']
    kk_segmentation[kk_segmentation == 4] = 3
    gray_matter = atropos['probability_images'][2]
    white_matter = (atropos['probability_images'][3] + atropos['probability_images'][4])
    kk = ants.kelly_kapowski(s=kk_segmentation, g=gray_matter, w=white_matter,
                            its=45, r=0.025, m=1.5, x=0, verbose=verbose)

    return_dict = {'thickness_image' : kk,
                   'csf_probability_image' : atropos['probability_images'][1],
                   'gray_matter_probability_image' : atropos['probability_images'][2],
                   'white_matter_probability_image' : atropos['probability_images'][3],
                   'deep_gray_matter_probability_image' : atropos['probability_images'][4],
                   'brain_stem_matter_probability_image' : atropos['probability_images'][5],
                   'cerebellum_probability_image' : atropos['probability_images'][6]
                  }
    return(return_dict)
