
import numpy as np
import ants

def claustrum_segmentation(t1,
                           do_preprocessing=True,
                           use_ensemble=True,
                           antsxnet_cache_directory=None,
                           verbose=False):

    """
    Claustrum segmentation

    Described here:

        https://arxiv.org/abs/2008.03465

    with the implementation available at:

        https://github.com/hongweilibran/claustrum_multi_view


    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image.

    do_preprocessing : boolean
        perform n4 bias correction.

    use_ensemble : boolean
        check whether to use all 3 sets of weights.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Claustrum segmentation probability image

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> probability_mask = claustrum_segmentation(image)
    """

    from ..architectures import create_sysu_media_unet_model_2d
    from ..utilities import brain_extraction
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    image_size = (180, 180)

    ################################
    #
    # Preprocess images
    #
    ################################

    number_of_channels = 1
    t1_preprocessed = ants.image_clone(t1)
    brain_mask = ants.threshold_image(t1, 0, 0, 0, 1)
    if do_preprocessing == True:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=True,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"]
        brain_mask = t1_preprocessing["brain_mask"]

    reference_image = ants.make_image((170, 256, 256),
                                       voxval=1,
                                       spacing=(1, 1, 1),
                                       origin=(0, 0, 0),
                                       direction=np.identity(3))
    center_of_mass_reference = ants.get_center_of_mass(reference_image)
    center_of_mass_image = ants.get_center_of_mass(brain_mask)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    t1_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, t1_preprocessed, reference_image)
    brain_mask_warped = ants.threshold_image(
        ants.apply_ants_transform_to_image(xfrm, brain_mask, reference_image), 0.5, 1.1, 1, 0 )

    ################################
    #
    # Gaussian normalize intensity based on brain mask
    #
    ################################

    mean_t1 = t1_preprocessed_warped[brain_mask_warped > 0].mean()
    std_t1 = t1_preprocessed_warped[brain_mask_warped > 0].std()
    t1_preprocessed_warped = (t1_preprocessed_warped - mean_t1) / std_t1

    t1_preprocessed_warped = t1_preprocessed_warped * brain_mask_warped

    ################################
    #
    # Build models and load weights
    #
    ################################

    number_of_models = 1
    if use_ensemble == True:
        number_of_models = 3

    if verbose == True:
        print("Claustrum:  retrieving axial model weights.")

    unet_axial_models = list()
    for i in range(number_of_models):
        weights_file_name = get_pretrained_network("claustrum_axial_" + str(i), antsxnet_cache_directory=antsxnet_cache_directory)
        unet_axial_models.append(create_sysu_media_unet_model_2d((*image_size, number_of_channels), anatomy="claustrum"))
        unet_axial_models[i].load_weights(weights_file_name)

    if verbose == True:
        print("Claustrum:  retrieving coronal model weights.")

    unet_coronal_models = list()
    for i in range(number_of_models):
        weights_file_name = get_pretrained_network("claustrum_coronal_" + str(i), antsxnet_cache_directory=antsxnet_cache_directory)
        unet_coronal_models.append(create_sysu_media_unet_model_2d((*image_size, number_of_channels), anatomy="claustrum"))
        unet_coronal_models[i].load_weights(weights_file_name)

    ################################
    #
    # Extract slices
    #
    ################################

    dimensions_to_predict = [1, 2]

    batch_coronal_X = np.zeros((t1_preprocessed_warped.shape[1], *image_size, number_of_channels))
    batch_axial_X = np.zeros((t1_preprocessed_warped.shape[2], *image_size, number_of_channels))

    for d in range(len(dimensions_to_predict)):
        number_of_slices = t1_preprocessed_warped.shape[dimensions_to_predict[d]]

        if verbose == True:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            t1_slice = pad_or_crop_image_to_size(ants.slice_image(t1_preprocessed_warped, dimensions_to_predict[d], i), image_size)
            if dimensions_to_predict[d] == 1:
                batch_coronal_X[i,:,:,0] = np.rot90(t1_slice.numpy(), k=-1)
            else:
                batch_axial_X[i,:,:,0] = np.rot90(t1_slice.numpy())

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    if verbose == True:
        print("Coronal prediction.")

    prediction_coronal = unet_coronal_models[0].predict(batch_coronal_X, verbose=verbose)
    if number_of_models > 1:
       for i in range(1, number_of_models, 1):
           prediction_coronal += unet_coronal_models[i].predict(batch_coronal_X, verbose=verbose)
    prediction_coronal /= number_of_models

    for i in range(t1_preprocessed_warped.shape[1]):
        prediction_coronal[i,:,:,0] = np.rot90(np.squeeze(prediction_coronal[i,:,:,0]))

    if verbose == True:
        print("Axial prediction.")

    prediction_axial = unet_axial_models[0].predict(batch_axial_X, verbose=verbose)
    if number_of_models > 1:
       for i in range(1, number_of_models, 1):
           prediction_axial += unet_axial_models[i].predict(batch_axial_X, verbose=verbose)
    prediction_axial /= number_of_models

    for i in range(t1_preprocessed_warped.shape[2]):
        prediction_axial[i,:,:,0] = np.rot90(np.squeeze(prediction_axial[i,:,:,0]), k=-1)

    if verbose == True:
        print("Restack image and transform back to native space.")

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    prediction_image_average = ants.image_clone(t1_preprocessed_warped) * 0

    for d in range(len(dimensions_to_predict)):
        which_batch_slices = range(t1_preprocessed_warped.shape[dimensions_to_predict[d]])
        prediction_per_dimension = None
        if dimensions_to_predict[d] == 1:
            prediction_per_dimension = prediction_coronal[which_batch_slices,:,:,:]
        else:
            prediction_per_dimension = prediction_axial[which_batch_slices,:,:,:]
        prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
        prediction_image = ants.copy_image_info(t1_preprocessed_warped,
          pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
            t1_preprocessed_warped.shape))
        prediction_image_average = prediction_image_average + (prediction_image - prediction_image_average) / (d + 1)

    probability_image = ants.apply_ants_transform_to_image(ants.invert_ants_transform(xfrm),
        prediction_image_average, t1) * ants.threshold_image(brain_mask, 0.5, 1, 1, 0)

    return(probability_image)
