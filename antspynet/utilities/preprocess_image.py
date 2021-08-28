import numpy as np
import tensorflow as tf
import ants

def preprocess_brain_image(image,
                           truncate_intensity=(0.01, 0.99),
                           brain_extraction_modality=None,
                           template_transform_type=None,
                           template="biobank",
                           do_bias_correction=True,
                           return_bias_field=False,
                           do_denoising=True,
                           intensity_matching_type=None,
                           reference_image=None,
                           intensity_normalization_type=None,
                           antsxnet_cache_directory=None,
                           verbose=True):

    """
    Basic preprocessing pipeline for T1-weighted brain MRI

    Standard preprocessing steps that have been previously described
    in various papers including the cortical thickness pipeline:

         https://www.ncbi.nlm.nih.gov/pubmed/24879923

    Arguments
    ---------
    image : ANTsImage
        input image

    truncate_intensity : 2-length tuple
        Defines the quantile threshold for truncating the image intensity

    brain_extraction_modality : string or None
        Perform brain extraction using antspynet tools.  One of "t1", "t1v0",
        "t1nobrainer", "t1combined", "flair", "t2", "bold", "fa", "t1infant",
        "t2infant", or None.

    template_transform_type : string
        See details in help for ants.registration.  Typically "Rigid" or
        "Affine".

    template : ANTs image (not skull-stripped)
        Alternatively, one can specify the default "biobank" or "croppedMni152"
        to download and use premade templates.

    do_bias_correction : boolean
        Perform N4 bias field correction.

    return_bias_field : boolean
        If True, return bias field as an additional output *without* bias
        correcting the preprocessed image.

    do_denoising : boolean
        Perform non-local means denoising.

    intensity_matching_type : string
        Either "regression" or "histogram". Only is performed if reference_image
        is not None.

    reference_image : ANTs image
        Reference image for intensity matching.

    intensity_normalization_type : string
        Either rescale the intensities to [0,1] (i.e., "01") or zero-mean, unit variance
        (i.e., "0mean").  If None normalization is not performed.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary with preprocessing information ANTs image (i.e., source_image) matched to the
    (reference_image).

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> preprocessed_image = preprocess_brain_image(image, do_brain_extraction=False)
    """

    from ..utilities import brain_extraction
    from ..utilities import regression_match_image
    from ..utilities import get_antsxnet_data

    preprocessed_image = ants.image_clone(image)

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    # Truncate intensity
    if truncate_intensity is not None:
        quantiles = (image.quantile(truncate_intensity[0]), image.quantile(truncate_intensity[1]))
        if verbose == True:
            print("Preprocessing:  truncate intensities ( low =", quantiles[0], ", high =", quantiles[1], ").")

        preprocessed_image[image < quantiles[0]] = quantiles[0]
        preprocessed_image[image > quantiles[1]] = quantiles[1]

    # Brain extraction
    mask = None
    if brain_extraction_modality is not None:
        if verbose == True:
            print("Preprocessing:  brain extraction.")

        probability_mask = brain_extraction(preprocessed_image, modality=brain_extraction_modality,
            antsxnet_cache_directory=antsxnet_cache_directory, verbose=verbose)
        mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        mask = ants.morphology(mask,"close",6).iMath_fill_holes()

    # Template normalization
    transforms = None
    if template_transform_type is not None:
        template_image = None
        if isinstance(template, str):
            template_file_name_path = get_antsxnet_data(template, antsxnet_cache_directory=antsxnet_cache_directory)
            template_image = ants.image_read(template_file_name_path)
        else:
            template_image = template

        if mask is None:
            registration = ants.registration(fixed=template_image, moving=preprocessed_image,
                type_of_transform=template_transform_type, verbose=verbose)
            preprocessed_image = registration['warpedmovout']
            transforms = dict(fwdtransforms=registration['fwdtransforms'],
                              invtransforms=registration['invtransforms'])
        else:
            template_probability_mask = brain_extraction(template_image, modality=brain_extraction_modality, 
                antsxnet_cache_directory=antsxnet_cache_directory, verbose=verbose)
            template_mask = ants.threshold_image(template_probability_mask, 0.5, 1, 1, 0)
            template_brain_image = template_mask * template_image

            preprocessed_brain_image = preprocessed_image * mask

            registration = ants.registration(fixed=template_brain_image, moving=preprocessed_brain_image,
                type_of_transform=template_transform_type, verbose=verbose)
            transforms = dict(fwdtransforms=registration['fwdtransforms'],
                              invtransforms=registration['invtransforms'])

            preprocessed_image = ants.apply_transforms(fixed = template_image, moving = preprocessed_image,
                transformlist=registration['fwdtransforms'], interpolator="linear", verbose=verbose)
            mask = ants.apply_transforms(fixed = template_image, moving = mask,
                transformlist=registration['fwdtransforms'], interpolator="genericLabel", verbose=verbose)

    # Do bias correction
    bias_field = None
    if do_bias_correction == True:
        if verbose == True:
            print("Preprocessing:  brain correction.")
        n4_output = None
        if mask is None:
            n4_output = ants.n4_bias_field_correction(preprocessed_image, shrink_factor=4, return_bias_field=return_bias_field, verbose=verbose)
        else:
            n4_output = ants.n4_bias_field_correction(preprocessed_image, mask, shrink_factor=4, return_bias_field=return_bias_field, verbose=verbose)
        if return_bias_field == True:
            bias_field = n4_output
        else:
            preprocessed_image = n4_output

    # Denoising
    if do_denoising == True:
        if verbose == True:
            print("Preprocessing:  denoising.")

        if mask is None:
            preprocessed_image = ants.denoise_image(preprocessed_image, shrink_factor=1)
        else:
            preprocessed_image = ants.denoise_image(preprocessed_image, mask, shrink_factor=1)

    # Image matching
    if reference_image is not None and intensity_matching_type is not None:
        if verbose == True:
            print("Preprocessing:  intensity matching.")

        if intensity_matching_type == "regression":
            preprocessed_image = regression_match_image(preprocessed_image, reference_image)
        elif intensity_matching_type == "histogram":
            preprocessed_image = ants.histogram_match_image(preprocessed_image, reference_image)
        else:
            raise ValueError("Unrecognized intensity_matching_type.")

    # Intensity normalization
    if intensity_normalization_type is not None:
        if verbose == True:
            print("Preprocessing:  intensity normalization.")

        if intensity_normalization_type == "01":
            preprocessed_image = (preprocessed_image - preprocessed_image.min())/(preprocessed_image.max() - preprocessed_image.min())
        elif intensity_normalization_type == "0mean":
            preprocessed_image = (preprocessed_image - preprocessed_image.mean())/preprocessed_image.std()
        else:
            raise ValueError("Unrecognized intensity_normalization_type.")

    return_dict = {'preprocessed_image' : preprocessed_image}
    if mask is not None:
        return_dict['brain_mask'] = mask
    if bias_field is not None:
        return_dict['bias_field'] = bias_field
    if transforms is not None:
        return_dict['template_transforms'] = transforms

    return(return_dict)
