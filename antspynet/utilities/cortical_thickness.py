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
    >>> image = ants.image_read("t1w_image.nii.gz")
    >>> kk = cortical_thickness(image)
    """

    from ..utilities import deep_atropos

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    atropos = deep_atropos(t1, do_preprocessing=True,
        antsxnet_cache_directory=antsxnet_cache_directory, verbose=True)

    # Kelly Kapowski cortical thickness

    kk_segmentation = ants.image_clone(atropos['segmentation_image'])
    kk_segmentation[kk_segmentation == 4] = 3
    gray_matter = atropos['probability_images'][2]
    white_matter = (atropos['probability_images'][3] + atropos['probability_images'][4])
    kk = ants.kelly_kapowski(s=kk_segmentation, g=gray_matter, w=white_matter,
                            its=45, r=0.025, m=1.5, x=0, verbose=int(verbose))

    return_dict = {'thickness_image' : kk,
                   'segmentation_image' : atropos['segmentation_image'],
                   'csf_probability_image' : atropos['probability_images'][1],
                   'gray_matter_probability_image' : atropos['probability_images'][2],
                   'white_matter_probability_image' : atropos['probability_images'][3],
                   'deep_gray_matter_probability_image' : atropos['probability_images'][4],
                   'brain_stem_probability_image' : atropos['probability_images'][5],
                   'cerebellum_probability_image' : atropos['probability_images'][6]
                  }
    return(return_dict)


def longitudinal_cortical_thickness(t1s,
                                    initial_template="oasis",
                                    number_of_iterations=1,
                                    refinement_transform="antsRegistrationSyNQuick[a]",
                                    antsxnet_cache_directory=None,
                                    verbose=False):

    """
    Perform KellyKapowski cortical thickness longitudinally using \code{deepAtropos}
    for segmentation of the derived single-subject template.  It takes inspiration from
    the work described here:

    https://pubmed.ncbi.nlm.nih.gov/31356207/

    Arguments
    ---------
    t1s : list of ANTsImage
        Input list of 3-D unprocessed t1-weighted brain images from a single subject.

    initial_template : string or ANTsImage
        Input image to define the orientation of the SST.  Can be a string (see
        get_antsxnet_data) or a specified template.  This allows the user to create a
        SST outside of this routine.

    number_of_iterations : int
        Defines the number of iterations for refining the SST.

    refinement_transform : string
       Transform for defining the refinement registration transform. See options in
       ants.registration.

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
    >>> t1s = list()
    >>> t1s.append(ants.image_read("t1w_image.nii.gz"))
    >>> kk = longitudinal_cortical_thickness(image)
    """

    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import deep_atropos

    ###################
    #
    #  Initial SST + optional affine refinement
    #
    ##################

    sst = None
    if isinstance(initial_template, str):
        template_file_name_path = get_antsxnet_data(initial_template, antsxnet_cache_directory=antsxnet_cache_directory)
        sst = ants.image_read(template_file_name_path)
    else:
        sst = initial_template

    for s in range(number_of_iterations):
        if verbose:
            print("Refinement iteration", s, "( out of", number_of_iterations, ")")

        sst_tmp = ants.image_clone(sst) * 0
        for i in range(len(t1s)):
            if verbose:
                print("***************************")
                print( "SST processing image", i, "( out of", len(t1s), ")")
                print( "***************************" )
            transform_type = "antsRegistrationSyNQuick[r]"
            if s > 0:
                transform_type = refinement_transform
            t1_preprocessed = preprocess_brain_image(t1s[i],
                truncate_intensity=(0.01, 0.99),
                brain_extraction_modality=None,
                template=sst,
                template_transform_type=transform_type,
                do_bias_correction=False,
                do_denoising=False,
                intensity_normalization_type="01",
                antsxnet_cache_directory=antsxnet_cache_directory,
                verbose=verbose)
            sst_tmp += t1_preprocessed['preprocessed_image']

        sst = sst_tmp / len(t1s)

    ###################
    #
    #  Preprocessing and affine transform to final SST
    #
    ##################

    t1s_preprocessed = list()
    for i in range(len(t1s)):
        if verbose:
            print("***************************")
            print( "Final processing image", i, "( out of", len(t1s), ")")
            print( "***************************" )
        t1_preprocessed = preprocess_brain_image(t1s[i],
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            template=sst,
            template_transform_type="antsRegistrationSyNQuick[a]",
            do_bias_correction=True,
            do_denoising=True,
            intensity_normalization_type="01",
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1s_preprocessed.append(t1_preprocessed)

    ###################
    #
    #  Deep  Atropos of SST for priors
    #
    ##################

    sst_atropos = deep_atropos(sst, do_preprocessing=True,
        antsxnet_cache_directory=antsxnet_cache_directory, verbose=verbose)

    ###################
    #
    #  Traditional Atropos + KK for each iamge
    #
    ##################

    return_list = list()
    for i in range(len(t1s_preprocessed)):
        if verbose:
            print("Atropos for image", i, "( out of", len(t1s), ")")
        atropos_output = ants.atropos(t1s_preprocessed[i]['preprocessed_image'],
            x=t1s_preprocessed[i]['brain_mask'], i=sst_atropos['probability_images'][1:7],
            m="[0.1,1x1x1]", c="[5,0]", priorweight=0.5, p="Socrates[1]", verbose=int(verbose))

        kk_segmentation = ants.image_clone(atropos_output['segmentation'])
        kk_segmentation[kk_segmentation == 4] = 3
        gray_matter = atropos_output['probabilityimages'][1]
        white_matter = atropos_output['probabilityimages'][2] + atropos_output['probabilityimages'][3]
        kk = ants.kelly_kapowski(s=kk_segmentation, g=gray_matter, w=white_matter,
            its=45, r=0.025, m=1.5, x=0, verbose=int(verbose))

        t1_dict = {'preprocessed_image' : t1s_preprocessed[i]['preprocessed_image'],
                   'thickness_image' : kk,
                   'segmentation_image' : atropos_output['segmentation'],
                   'csf_probability_image' : atropos_output['probabilityimages'][0],
                   'gray_matter_probability_image' : atropos_output['probabilityimages'][1],
                   'white_matter_probability_image' : atropos_output['probabilityimages'][2],
                   'deep_gray_matter_probability_image' : atropos_output['probabilityimages'][3],
                   'brain_stem_probability_image' : atropos_output['probabilityimages'][4],
                   'cerebellum_probability_image' : atropos_output['probabilityimages'][5],
                   'template_transforms' : t1s_preprocessed[i]['template_transforms']
                  }
        return_list.append(t1_dict)

    return_list.append(sst)

    return(return_list)

