
import ants
import numpy as np
from tensorflow import keras

def sysu_media_wmh_segmentation(flair,
                                t1=None,
                                use_ensemble=True,
                                antsxnet_cache_directory=None,
                                verbose=False):

    """
    Perform WMH segmentation using the winning submission in the MICCAI
    2017 challenge by the sysu_media team using FLAIR or T1/FLAIR.  The
    MICCAI challenge is discussed in

    https://pubmed.ncbi.nlm.nih.gov/30908194/

    with the sysu_media's team entry is discussed in

     https://pubmed.ncbi.nlm.nih.gov/30125711/

    with the original implementation available here:

    https://github.com/hongweilibran/wmh_ibbmTum

    The original implementation used global thresholding as a quick
    brain extraction approach.  Due to possible generalization difficulties,
    we leave such post-processing steps to the user.  For brain or white
    matter masking see functions brain_extraction or deep_atropos,
    respectively.

    Arguments
    ---------
    flair : ANTsImage
        input 3-D FLAIR brain image (not skull-stripped).

    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    use_ensemble : boolean
        check whether to use all 3 sets of weights.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> image = ants.image_read("flair.nii.gz")
    >>> probability_mask = sysu_media_wmh_segmentation(image)
    """

    from ..architectures import create_sysu_media_unet_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import preprocess_brain_image
    from ..utilities import binary_dice_coefficient

    if flair.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    image_size = (200, 200)

    ################################
    #
    # Preprocess images
    #
    ################################

    def closest_simplified_direction_matrix(direction):
        closest = (np.abs(direction) + 0.5).astype(int).astype(float)
        closest[direction < 0] *= -1.0
        return closest

    simplified_direction = closest_simplified_direction_matrix(flair.direction)

    flair_preprocessing = preprocess_brain_image(flair,
        truncate_intensity=None,
        brain_extraction_modality=None,
        do_bias_correction=False,
        do_denoising=False,
        antsxnet_cache_directory=antsxnet_cache_directory,
        verbose=verbose)
    flair_preprocessed = flair_preprocessing["preprocessed_image"]
    flair_preprocessed.set_direction(simplified_direction)
    flair_preprocessed.set_origin((0, 0, 0))
    flair_preprocessed.set_spacing((1, 1, 1))
    number_of_channels = 1

    t1_preprocessed = None
    if t1 is not None:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality=None,
            do_bias_correction=False,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"]
        t1_preprocessed.set_direction(simplified_direction)
        t1_preprocessed.set_origin((0, 0, 0))
        t1_preprocessed.set_spacing((1, 1, 1))
        number_of_channels = 2

    ################################
    #
    # Reorient images
    #
    ################################

    reference_image = ants.make_image((256, 256, 256),
                                       voxval=0,
                                       spacing=(1, 1, 1),
                                       origin=(0, 0, 0),
                                       direction=np.identity(3))
    center_of_mass_reference = np.floor(ants.get_center_of_mass(reference_image * 0 + 1))
    center_of_mass_image = np.floor(ants.get_center_of_mass(flair_preprocessed))
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    flair_preprocessed_warped = ants.apply_ants_transform_to_image(
        xfrm, flair_preprocessed, reference_image, interpolation="nearestneighbor")
    crop_image = ants.image_clone(flair_preprocessed) * 0 + 1
    crop_image_warped = ants.apply_ants_transform_to_image(
        xfrm, crop_image, reference_image, interpolation="nearestneighbor")
    flair_preprocessed_warped = ants.crop_image(flair_preprocessed_warped, crop_image_warped, 1)

    if t1 is not None:
        t1_preprocessed_warped = ants.apply_ants_transform_to_image(
            xfrm, t1_preprocessed, reference_image, interpolation="nearestneighbor")
        t1_preprocessed_warped = ants.crop_image(t1_preprocessed_warped, crop_image_warped, 1)

    ################################
    #
    # Gaussian normalize intensity
    #
    ################################

    mean_flair = flair_preprocessed.mean()
    std_flair = flair_preprocessed.std()
    if number_of_channels == 2:
        mean_t1 = t1_preprocessed.mean()
        std_t1 = t1_preprocessed.std()

    flair_preprocessed_warped = (flair_preprocessed_warped - mean_flair) / std_flair
    if number_of_channels == 2:
        t1_preprocessed_warped = (t1_preprocessed_warped - mean_t1) / std_t1

    ################################
    #
    # Build models and load weights
    #
    ################################

    number_of_models = 1
    if use_ensemble:
        number_of_models = 3

    if verbose:
        print("White matter hyperintensity:  retrieving model weights.")

    unet_models = list()
    for i in range(number_of_models):
        if number_of_channels == 1:
            weights_file_name = get_pretrained_network("sysuMediaWmhFlairOnlyModel" + str(i),
                antsxnet_cache_directory=antsxnet_cache_directory)
        else:
            weights_file_name = get_pretrained_network("sysuMediaWmhFlairT1Model" + str(i),
                antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model = create_sysu_media_unet_model_2d((*image_size, number_of_channels))
        unet_loss = binary_dice_coefficient(smoothing_factor=1.)
        unet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                        loss=unet_loss)
        unet_model.load_weights(weights_file_name)
        unet_models.append(unet_model)


    ################################
    #
    # Extract slices
    #
    ################################

    dimensions_to_predict = [2]

    total_number_of_slices = 0
    for d in range(len(dimensions_to_predict)):
        total_number_of_slices += flair_preprocessed_warped.shape[dimensions_to_predict[d]]

    batchX = np.zeros((total_number_of_slices, *image_size, number_of_channels))

    slice_count = 0
    for d in range(len(dimensions_to_predict)):
        number_of_slices = flair_preprocessed_warped.shape[dimensions_to_predict[d]]

        if verbose:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            flair_slice = pad_or_crop_image_to_size(ants.slice_image(flair_preprocessed_warped, dimensions_to_predict[d], i), image_size)
            batchX[slice_count,:,:,0] = flair_slice.numpy()
            if number_of_channels == 2:
                t1_slice = pad_or_crop_image_to_size(ants.slice_image(t1_preprocessed_warped, dimensions_to_predict[d], i), image_size)
                batchX[slice_count,:,:,1] = t1_slice.numpy()
            slice_count += 1

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    if verbose:
        print("Prediction.")

    prediction = unet_models[0].predict(np.transpose(batchX, axes=(0, 2, 1, 3)), verbose=verbose)
    if number_of_models > 1:
        for i in range(1, number_of_models, 1):
            prediction += unet_models[i].predict(np.transpose(batchX, axes=(0, 2, 1, 3)), verbose=verbose)
    prediction /= number_of_models
    prediction = np.transpose(prediction, axes=(0, 2, 1, 3))

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    prediction_image_average = ants.image_clone(flair_preprocessed_warped) * 0

    current_start_slice = 0
    for d in range(len(dimensions_to_predict)):
        current_end_slice = current_start_slice + flair_preprocessed_warped.shape[dimensions_to_predict[d]]
        which_batch_slices = range(current_start_slice, current_end_slice)
        prediction_per_dimension = prediction[which_batch_slices,:,:,:]
        prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
        prediction_image = ants.copy_image_info(flair_preprocessed_warped,
          pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
            flair_preprocessed_warped.shape))
        prediction_image_average = prediction_image_average + (prediction_image - prediction_image_average) / (d + 1)
        current_start_slice = current_end_slice

    probability_image = ants.apply_ants_transform_to_image(
        ants.invert_ants_transform(xfrm), prediction_image_average, flair_preprocessed)
    probability_image = ants.copy_image_info(flair, probability_image)

    return(probability_image)

def hypermapp3r_segmentation(t1,
                             flair,
                             number_of_monte_carlo_iterations=30,
                             do_preprocessing=True,
                             antsxnet_cache_directory=None,
                             verbose=False):

    """
    Perform HyperMapp3r (white matter hyperintensities) segmentation described in

    https://pubmed.ncbi.nlm.nih.gov/35088930/

    with models and architecture ported from

    https://github.com/mgoubran/HyperMapp3r

    Additional documentation and attribution resources found at

    https://hypermapp3r.readthedocs.io/en/latest/

    Preprocessing consists of:
       * n4 bias correction and
       * brain extraction
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D t1-weighted MR image.  Assumed to be aligned with the flair.

    flair : ANTsImage
        input 3-D flair MR image.  Assumed to be aligned with the t1.

    do_preprocessing : boolean
        See description above.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    ANTs labeled wmh segmentationimage.

    Example
    -------
    >>> mask = hypermapp3r_segmentation(t1, flair)
    """

    from ..architectures import create_hypermapp3r_unet_model_3d
    from ..utilities import preprocess_brain_image
    from ..utilities import get_pretrained_network

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    ################################
    #
    # Preprocess images
    #
    ################################

    if verbose:
        print("*************  Preprocessing  ***************")
        print("")

    t1_preprocessed = t1
    brain_mask = None
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        brain_mask = t1_preprocessing['brain_mask']
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask
    else:
        # If we don't generate the mask from the preprocessing, we assume that we
        # can extract the brain directly from the foreground of the t1 image.
        brain_mask = ants.threshold_image(t1, 0, 0, 0, 1)

    t1_preprocessed_mean = t1_preprocessed[brain_mask > 0].mean()
    t1_preprocessed_std = t1_preprocessed[brain_mask > 0].std()
    t1_preprocessed[brain_mask > 0] = (t1_preprocessed[brain_mask > 0] - t1_preprocessed_mean) / t1_preprocessed_std

    flair_preprocessed = flair
    if do_preprocessing:
        flair_preprocessing = preprocess_brain_image(flair,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality=None,
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        flair_preprocessed = flair_preprocessing["preprocessed_image"] * brain_mask

    flair_preprocessed_mean = flair_preprocessed[brain_mask > 0].mean()
    flair_preprocessed_std = flair_preprocessed[brain_mask > 0].std()
    flair_preprocessed[brain_mask > 0] = (flair_preprocessed[brain_mask > 0] - flair_preprocessed_mean) / flair_preprocessed_std

    if verbose:
        print("    HyperMapp3r: reorient input images.")

    channel_size = 2
    input_image_size = (224, 224, 224)
    template_array = np.ones(input_image_size)
    template_direction = np.eye(3)
    template_direction[1, 1] = -1.0
    reorient_template = ants.from_numpy(template_array, origin=(0, 0, 0), spacing=(1, 1, 1),
        direction=template_direction)

    center_of_mass_template = ants.get_center_of_mass(reorient_template)
    center_of_mass_image = ants.get_center_of_mass(brain_mask)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)

    batchX = np.zeros((1, *input_image_size, channel_size))

    t1_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, t1_preprocessed, reorient_template)
    batchX[0,:,:,:,0] = t1_preprocessed_warped.numpy()

    flair_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, flair_preprocessed, reorient_template)
    batchX[0,:,:,:,1] = flair_preprocessed_warped.numpy()

    if verbose:
        print("    HyperMapp3r: generate network and load weights.")

    model = create_hypermapp3r_unet_model_3d((*input_image_size, 2))
    weights_file_name = get_pretrained_network("hyperMapp3r", antsxnet_cache_directory=antsxnet_cache_directory)
    model.load_weights(weights_file_name)

    if verbose:
        print("    HyperMapp3r: prediction.")

    if verbose:
        print("    HyperMapp3r: Monte Carlo iterations (SpatialDropout).")

    prediction_array = np.zeros(input_image_size)
    for i in range(number_of_monte_carlo_iterations):
        if verbose:
            print("        Monte Carlo iteration", i + 1, "out of", number_of_monte_carlo_iterations)
        prediction_array = (np.squeeze(model.predict(batchX, verbose=verbose)) + i * prediction_array) / (i + 1)

    prediction_image = ants.from_numpy(prediction_array, origin=reorient_template.origin,
        spacing=reorient_template.spacing, direction=reorient_template.direction)

    xfrm_inv = xfrm.invert()
    probability_image = xfrm_inv.apply_to_image(prediction_image, t1)
    return(probability_image)

def wmh_segmentation(flair,
                     t1,
                     white_matter_mask=None,
                     use_combined_model=True,
                     prediction_batch_size=16,
                     do_preprocessing=True,
                     antsxnet_cache_directory=None,
                     verbose=False):

    """
    Perform White matter hyperintensity probabilistic segmentation
    given a pre-aligned FLAIR and T2 images.  Note that the underlying
    model is 3-D and requires images to be of > 64 voxels in each
    dimension.

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * brain extraction

    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    \code{do_preprocessing = True}

    Arguments
    ---------
    flair : ANTsImage
        input 3-D FLAIR brain image (not skull-stripped).

    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    white_matter_mask : ANTsImage
        input white matter mask for patch extraction. If None, calculated using
        deep_atropos (labels 3 and 4).

    use_combined_model : boolean
        Original or combined.

    prediction_batch_size : int
        Control memory usage for prediction.  More consequential for GPU-usage.

    do_preprocessing : boolean
        perform n4 bias correction, intensity truncation, brain extraction.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> flair = ants.image_read("flair.nii.gz")
    >>> t1 = ants.image_read("t1.nii.gz")
    >>> probability_mask = wmh_segmentation(flair, t1)
    """

    from ..architectures import create_sysu_media_unet_model_3d
    from ..utilities import deep_atropos
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image

    if np.any(t1.shape < np.array((64, 64, 64))):
        raise ValueError("Images must be > 64 voxels per dimension.")

    ################################
    #
    # Preprocess images
    #
    ################################

    if white_matter_mask is None:
        if verbose:
            print("Calculate white matter mask.")
        atropos = deep_atropos(t1, do_preprocessing=True, verbose=verbose)
        white_matter_mask = ants.threshold_image(atropos['segmentation_image'], 3, 4, 1, 0)

    t1_preprocessed = None
    flair_preprocessed = None

    if do_preprocessing:
        if verbose:
            print("Preprocess T1 and FLAIR images.")

        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.995),
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        brain_mask = ants.threshold_image(t1_preprocessing["brain_mask"], 0.5, 1, 1, 0)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask

        flair_preprocessing = preprocess_brain_image(flair,
            truncate_intensity=None,
            brain_extraction_modality=None,
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        flair_preprocessed = flair_preprocessing["preprocessed_image"] * brain_mask

    else:
        t1_preprocessed = ants.image_clone(t1)
        flair_preprocessed = ants.image_clone(flair)

    white_matter_indices = white_matter_mask > 0
    t1_preprocessed_min = t1_preprocessed[white_matter_indices].min()
    t1_preprocessed_max = t1_preprocessed[white_matter_indices].max()
    flair_preprocessed_min = flair_preprocessed[white_matter_indices].min()
    flair_preprocessed_max = flair_preprocessed[white_matter_indices].max()

    t1_preprocessed = (t1_preprocessed - t1_preprocessed_min) / (t1_preprocessed_max - t1_preprocessed_min)
    flair_preprocessed = (flair_preprocessed - flair_preprocessed_min) / (flair_preprocessed_max - flair_preprocessed_min)

    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Load model and weights.")

    patch_size = (64, 64, 64)
    stride_length = (32, 32, 32)
    number_of_filters = (64, 96, 128, 256, 512)
    channel_size = 2

    model = create_sysu_media_unet_model_3d((*patch_size, channel_size),
                                             number_of_filters=number_of_filters)
    weights_file_name = None
    if use_combined_model:
        weights_file_name = get_pretrained_network("antsxnetWmhOr", antsxnet_cache_directory=antsxnet_cache_directory)
    else:
        weights_file_name = get_pretrained_network("antsxnetWmh", antsxnet_cache_directory=antsxnet_cache_directory)
    model.load_weights(weights_file_name)

    ################################
    #
    # Extract patches
    #
    ################################

    if verbose:
        print("Extract patches.")

    t1_patches = extract_image_patches(t1_preprocessed,
                                       patch_size=patch_size,
                                       max_number_of_patches="all",
                                       stride_length=stride_length,
                                       mask_image=white_matter_mask,
                                       random_seed=None,
                                       return_as_array=True)
    flair_patches = extract_image_patches(flair_preprocessed,
                                          patch_size=patch_size,
                                          max_number_of_patches="all",
                                          stride_length=stride_length,
                                          mask_image=white_matter_mask,
                                          random_seed=None,
                                          return_as_array=True)

    total_number_of_patches = t1_patches.shape[0]

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    number_of_full_batches = total_number_of_patches // prediction_batch_size
    if verbose:
        print("Total number of patches: ", str(total_number_of_patches))
        print("Prediction batch size: ", str(prediction_batch_size))
        print("Number of batches: ", str(number_of_full_batches + 1))
     
    prediction = np.zeros((total_number_of_patches, *patch_size, 1))
    for b in range(number_of_full_batches + 1):
        batchX = None
        if b < number_of_full_batches:
            batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
        else:
            residual_number_of_patches = total_number_of_patches - number_of_full_batches * prediction_batch_size
            batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

        indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
        batchX[:,:,:,:,0] = flair_patches[indices,:,:,:]
        batchX[:,:,:,:,1] = t1_patches[indices,:,:,:]
        
        if verbose:
            print("Predicting batch ", str(b + 1), " of ", str(number_of_full_batches + 1))  
        prediction[indices,:,:,:,:] = model.predict(batchX, verbose=verbose)

    if verbose:
        print("Predict patches and reconstruct.")


    wmh_probability_image = reconstruct_image_from_patches(np.squeeze(prediction),
                                                           stride_length=stride_length,
                                                           domain_image=white_matter_mask,
                                                           domain_image_is_mask=True)

    return(wmh_probability_image)
