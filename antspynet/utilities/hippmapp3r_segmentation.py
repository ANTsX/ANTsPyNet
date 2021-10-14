import numpy as np
import tensorflow as tf
import ants

def hippmapp3r_segmentation(t1,
                            do_preprocessing=True,
                            antsxnet_cache_directory=None,
                            verbose=False):

    """
    Perform HippMapp3r (hippocampal) segmentation described in

     https://www.ncbi.nlm.nih.gov/pubmed/31609046

    with models and architecture ported from

    https://github.com/mgoubran/HippMapp3r

    Additional documentation and attribution resources found at

    https://hippmapp3r.readthedocs.io/en/latest/

    Preprocessing consists of:
       * n4 bias correction and
       * brain extraction
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        input image

    do_preprocessing : boolean
        See description above.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    ANTs labeled hippocampal image.

    Example
    -------
    >>> mask = hippmapp3r_segmentation(t1)
    """

    from ..architectures import create_hippmapp3r_unet_model_3d
    from ..utilities import preprocess_brain_image
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if verbose == True:
        print("*************  Preprocessing  ***************")
        print("")

    t1_preprocessed = t1
    if do_preprocessing == True:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1",
            template=None,
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

    if verbose == True:
        print("*************  Initial stage segmentation  ***************")
        print("")

    # Normalize to mprage_hippmapp3r space
    if verbose == True:
        print("    HippMapp3r: template normalization.")

    template_file_name_path = get_antsxnet_data("mprage_hippmapp3r", antsxnet_cache_directory=antsxnet_cache_directory)
    template_image = ants.image_read(template_file_name_path)

    registration = ants.registration(fixed=template_image, moving=t1_preprocessed,
        type_of_transform="antsRegistrationSyNQuickRepro[t]", verbose=verbose)
    image = registration['warpedmovout']
    transforms = dict(fwdtransforms=registration['fwdtransforms'],
                        invtransforms=registration['invtransforms'])

    # Threshold at 10th percentile of non-zero voxels in "robust range (fslmaths)"
    if verbose == True:
        print("    HippMapp3r: threshold.")

    image_array = image.numpy()
    image_robust_range = np.quantile(image_array[np.where(image_array != 0)], (0.02, 0.98))
    threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]
    thresholded_mask = ants.threshold_image(image, -10000, threshold_value, 0, 1)
    thresholded_image = image * thresholded_mask

    # Standardize image
    if verbose == True:
        print("    HippMapp3r: standardize.")

    mean_image = np.mean(thresholded_image[thresholded_mask==1])
    sd_image = np.std(thresholded_image[thresholded_mask==1])
    image_normalized = (image - mean_image) / sd_image
    image_normalized = image_normalized * thresholded_mask

    # Trim and resample image
    if verbose == True:
        print("    HippMapp3r: trim and resample to (160, 160, 128).")

    image_cropped = ants.crop_image(image_normalized, thresholded_mask, 1)
    shape_initial_stage = (160, 160, 128)
    image_resampled = ants.resample_image(image_cropped, shape_initial_stage, use_voxels=True, interp_type=1)

    if verbose == True:
        print("    HippMapp3r: generate first network and download weights.")

    model_initial_stage = create_hippmapp3r_unet_model_3d((*shape_initial_stage, 1), do_first_network=True)

    initial_stage_weights_file_name = get_pretrained_network("hippMapp3rInitial", antsxnet_cache_directory=antsxnet_cache_directory)
    model_initial_stage.load_weights(initial_stage_weights_file_name)

    if verbose == True:
        print("    HippMapp3r: prediction.")

    data_initial_stage = np.expand_dims(image_resampled.numpy(), axis=0)
    data_initial_stage = np.expand_dims(data_initial_stage, axis=-1)
    mask_array = model_initial_stage.predict(data_initial_stage, verbose=verbose)
    mask_image_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(np.squeeze(mask_array)))
    mask_image = ants.resample_image(mask_image_resampled, image.shape, use_voxels=True, interp_type=0)
    mask_image[mask_image >= 0.5] = 1
    mask_image[mask_image < 0.5] = 0

    #########################################
    #
    # Perform refined (stage 2) segmentation
    #

    if verbose == True:
        print("")
        print("")
        print("*************  Refine stage segmentation  ***************")
        print("")

    mask_array = np.squeeze(mask_array)
    centroid_indices = np.where(mask_array == 1)
    centroid = np.zeros((3,))
    centroid[0] = centroid_indices[0].mean()
    centroid[1] = centroid_indices[1].mean()
    centroid[2] = centroid_indices[2].mean()

    shape_refine_stage = (112, 112, 64)
    lower = (np.floor(centroid - 0.5 * np.array(shape_refine_stage)) - 1).astype(int)
    upper = (lower + np.array(shape_refine_stage)).astype(int)

    image_trimmed = ants.crop_indices(image_resampled, lower.astype(int), upper.astype(int))

    if verbose == True:
        print("    HippMapp3r: generate second network and download weights.")

    model_refine_stage = create_hippmapp3r_unet_model_3d((*shape_refine_stage, 1), do_first_network=False)

    refine_stage_weights_file_name = get_pretrained_network("hippMapp3rRefine", antsxnet_cache_directory=antsxnet_cache_directory)
    model_refine_stage.load_weights(refine_stage_weights_file_name)

    data_refine_stage = np.expand_dims(image_trimmed.numpy(), axis=0)
    data_refine_stage = np.expand_dims(data_refine_stage, axis=-1)

    if verbose == True:
        print("    HippMapp3r: Monte Carlo iterations (SpatialDropout).")

    number_of_mci_iterations = 30
    prediction_refine_stage = np.zeros(shape_refine_stage)
    for i in range(number_of_mci_iterations):
        tf.random.set_seed(i)
        if verbose == True:
            print("        Monte Carlo iteration", i + 1, "out of", number_of_mci_iterations)
        prediction_refine_stage = \
            (np.squeeze(model_refine_stage.predict(data_refine_stage, verbose=verbose)) + \
             i * prediction_refine_stage ) / (i + 1)

    prediction_refine_stage_array = np.zeros(image_resampled.shape)
    prediction_refine_stage_array[lower[0]:upper[0],
                                  lower[1]:upper[1],
                                  lower[2]:upper[2]] = prediction_refine_stage
    probability_mask_refine_stage_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(prediction_refine_stage_array))

    segmentation_image_resampled = ants.label_clusters(
        ants.threshold_image(probability_mask_refine_stage_resampled, 0.0, 0.5, 0, 1), min_cluster_size=10)
    segmentation_image_resampled[segmentation_image_resampled > 2] = 0
    geom = ants.label_geometry_measures(segmentation_image_resampled)
    if len(geom['VolumeInMillimeters']) < 2:
        raise ValueError("Error: left and right hippocampus not found.")

    if geom['Centroid_x'][0] < geom['Centroid_x'][1]:
        segmentation_image_resampled[segmentation_image_resampled == 1] = 3
        segmentation_image_resampled[segmentation_image_resampled == 2] = 1
        segmentation_image_resampled[segmentation_image_resampled == 3] = 2

    segmentation_image = ants.apply_transforms(fixed=t1,
      moving=segmentation_image_resampled, transformlist=transforms['invtransforms'],
      whichtoinvert=[True], interpolator="genericLabel", verbose=verbose)

    return(segmentation_image)
