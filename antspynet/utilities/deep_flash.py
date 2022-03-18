import numpy as np
import ants

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def deep_flash(t1,
                t2=None,
                do_preprocessing=True,
                use_rank_intensity=True,
                antsxnet_cache_directory=None,
                verbose=False
                ):

    """
    Hippocampal/Enthorhinal segmentation using "Deep Flash"

    Perform hippocampal/entorhinal segmentation in T1 images using
    labels from Mike Yassa's lab

    https://faculty.sites.uci.edu/myassa/

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA2/CA3/CA4
    Label 14:  right DG/CA2/CA3/CA4
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * affine registration to the "deep flash" template.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    t2 : ANTsImage
        Optional 3-D T2-weighted brain image.  If specified, it is assumed to be
        pre-aligned to the t1.

    do_preprocessing : boolean
        See description above.

    use_rank_intensity : boolean
        If false, use histogram matching with cropped template ROI.  Otherwise,
        use a rank intensity transform on the cropped ROI.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label and foreground.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_flash(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import brain_extraction

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    ################################
    #
    # Options temporarily taken from the user
    #
    ################################

    # use_hierarchical_parcellation : boolean
    #     If True, use u-net model with additional outputs of the medial temporal lobe
    #     region, hippocampal, and entorhinal/perirhinal/parahippocampal regions.  Otherwise
    #     the only additional output is the medial temporal lobe.
    #
    # use_contralaterality : boolean
    #     Use both hemispherical models to also predict the corresponding contralateral
    #     segmentation and use both sets of priors to produce the results.  Mainly used
    #     for debugging.

    use_hierarchical_parcellation = True
    use_contralaterality = True

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = t1
    t1_mask = None
    t1_preprocessed_flipped = None
    t1_template = ants.image_read(get_antsxnet_data("deepFlashTemplateT1SkullStripped"))
    template_transforms = None
    if do_preprocessing:

        if verbose == True:
            print("Preprocessing T1.")

        # Brain extraction
        probability_mask = brain_extraction(t1_preprocessed, modality="t1",
            antsxnet_cache_directory=antsxnet_cache_directory, verbose=verbose)
        t1_mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        t1_preprocessed = t1_preprocessed * t1_mask

        # Do bias correction
        t1_preprocessed = ants.n4_bias_field_correction(t1_preprocessed, t1_mask, shrink_factor=4, verbose=verbose)

        # Warp to template
        registration = ants.registration(fixed=t1_template, moving=t1_preprocessed,
            type_of_transform="antsRegistrationSyNQuickRepro[a]", verbose=verbose)
        template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                                   invtransforms=registration['invtransforms'])
        t1_preprocessed = registration['warpedmovout']

    if use_contralaterality:
        t1_preprocessed_array = t1_preprocessed.numpy()
        t1_preprocessed_array_flipped = np.flip(t1_preprocessed_array, axis=0)
        t1_preprocessed_flipped = ants.from_numpy(t1_preprocessed_array_flipped,
                                                    origin=t1_preprocessed.origin,
                                                    spacing=t1_preprocessed.spacing,
                                                    direction=t1_preprocessed.direction)

    t2_preprocessed = t2
    t2_preprocessed_flipped = None
    t2_template = None
    if t2 is not None:
        t2_template = ants.image_read(get_antsxnet_data("deepFlashTemplateT2SkullStripped"))
        t2_template = ants.copy_image_info(t1_template, t2_template)
        if do_preprocessing:

            if verbose == True:
                print("Preprocessing T2.")

            # Brain extraction
            t2_preprocessed = t2_preprocessed * t1_mask

            # Do bias correction
            t2_preprocessed = ants.n4_bias_field_correction(t2_preprocessed, t1_mask, shrink_factor=4, verbose=verbose)

            # Warp to template
            t2_preprocessed = ants.apply_transforms(fixed=t1_template,
                moving=t2_preprocessed, transformlist=template_transforms['fwdtransforms'],
                verbose=verbose)

        if use_contralaterality:
            t2_preprocessed_array = t2_preprocessed.numpy()
            t2_preprocessed_array_flipped = np.flip(t2_preprocessed_array, axis=0)
            t2_preprocessed_flipped = ants.from_numpy(t2_preprocessed_array_flipped,
                                                    origin=t2_preprocessed.origin,
                                                    spacing=t2_preprocessed.spacing,
                                                    direction=t2_preprocessed.direction)

    probability_images = list()
    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
    image_size = (64, 64, 96)

    ################################
    #
    # Process left/right in split networks
    #
    ################################

    ################################
    #
    # Download spatial priors
    #
    ################################

    spatial_priors_file_name_path = get_antsxnet_data("deepFlashPriors",
        antsxnet_cache_directory=antsxnet_cache_directory)
    spatial_priors = ants.image_read(spatial_priors_file_name_path)
    priors_image_list = ants.ndimage_to_list(spatial_priors)
    for i in range(len(priors_image_list)):
        priors_image_list[i] = ants.copy_image_info(t1_preprocessed, priors_image_list[i])

    labels_left = labels[1::2]
    priors_image_left_list = priors_image_list[1::2]
    probability_images_left = list()
    foreground_probability_images_left = list()
    lower_bound_left = (76, 74, 56)
    upper_bound_left = (140, 138, 152)
    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    origin_left = tmp_cropped.origin

    spacing = tmp_cropped.spacing
    direction = tmp_cropped.direction

    t1_template_roi_left = ants.crop_indices(t1_template, lower_bound_left, upper_bound_left)
    t1_template_roi_left = (t1_template_roi_left - t1_template_roi_left.min()) / (t1_template_roi_left.max() - t1_template_roi_left.min()) * 2.0 - 1.0
    t2_template_roi_left = None
    if t2_template is not None:
        t2_template_roi_left = ants.crop_indices(t2_template, lower_bound_left, upper_bound_left)
        t2_template_roi_left = (t2_template_roi_left - t2_template_roi_left.min()) / (t2_template_roi_left.max() - t2_template_roi_left.min()) * 2.0 - 1.0

    labels_right = labels[2::2]
    priors_image_right_list = priors_image_list[2::2]
    probability_images_right = list()
    foreground_probability_images_right = list()
    lower_bound_right = (20, 74, 56)
    upper_bound_right = (84, 138, 152)
    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    origin_right = tmp_cropped.origin

    t1_template_roi_right = ants.crop_indices(t1_template, lower_bound_right, upper_bound_right)
    t1_template_roi_right = (t1_template_roi_right - t1_template_roi_right.min()) / (t1_template_roi_right.max() - t1_template_roi_right.min()) * 2.0 - 1.0
    t2_template_roi_right = None
    if t2_template is not None:
        t2_template_roi_right = ants.crop_indices(t2_template, lower_bound_right, upper_bound_right)
        t2_template_roi_right = (t2_template_roi_right - t2_template_roi_right.min()) / (t2_template_roi_right.max() - t2_template_roi_right.min()) * 2.0 - 1.0


    ################################
    #
    # Create model
    #
    ################################

    channel_size = 1 + len(labels_left)
    if t2 is not None:
        channel_size += 1

    number_of_classification_labels = 1 + len(labels_left)

    unet_model = create_unet_model_3d((*image_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0, weight_decay=0)

    penultimate_layer = unet_model.layers[-2].output

    # medial temporal lobe
    output1 = Conv3D(filters=1,
                     kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

    if use_hierarchical_parcellation:

        # EC, perirhinal, and parahippo.
        output2 = Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

        # Hippocampus
        output3 = Conv3D(filters=1,
                        kernel_size=(1, 1, 1),
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

        unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output1, output2, output3])
    else:
        unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output1])

    ################################
    #
    # Left:  build model and load weights
    #
    ################################

    network_name = 'deepFlashLeftT1'
    if t2 is not None:
        network_name = 'deepFlashLeftBoth'

    if use_hierarchical_parcellation:
        network_name += "Hierarchical"

    if use_rank_intensity:
        network_name += "_ri"

    if verbose:
        print("DeepFlash: retrieving model weights (left).")
    weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Left:  do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Prediction (left).")

    batchX = None
    if use_contralaterality:
        batchX = np.zeros((2, *image_size, channel_size))
    else:
        batchX = np.zeros((1, *image_size, channel_size))

    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    if use_rank_intensity:
        t1_cropped = ants.rank_intensity(t1_cropped)
    else:
        t1_cropped = ants.histogram_match_image(t1_cropped, t1_template_roi_left, 255, 64, False)
    batchX[0,:,:,:,0] = t1_cropped.numpy()
    if use_contralaterality:
        t1_cropped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_left, upper_bound_left)
        if use_rank_intensity:
            t1_cropped = ants.rank_intensity(t1_cropped)
        else:
            t1_cropped = ants.histogram_match_image(t1_cropped, t1_template_roi_left, 255, 64, False)
        batchX[1,:,:,:,0] = t1_cropped.numpy()
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_left, upper_bound_left)
        if use_rank_intensity:
            t2_cropped = ants.rank_intensity(t2_cropped)
        else:
            t2_cropped = ants.histogram_match_image(t2_cropped, t2_template_roi_left, 255, 64, False)
        batchX[0,:,:,:,1] = t2_cropped.numpy()
        if use_contralaterality:
            t2_cropped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_left, upper_bound_left)
            if use_rank_intensity:
                t2_cropped = ants.rank_intensity(t2_cropped)
            else:
                t2_cropped = ants.histogram_match_image(t2_cropped, t2_template_roi_left, 255, 64, False)
            batchX[1,:,:,:,1] = t2_cropped.numpy()

    for i in range(len(priors_image_left_list)):
        cropped_prior = ants.crop_indices(priors_image_left_list[i], lower_bound_left, upper_bound_left)
        for j in range(batchX.shape[0]):
            batchX[j,:,:,:,i + (channel_size - len(labels_left))] = cropped_prior.numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    for i in range(1 + len(labels_left)):
        for j in range(predicted_data[0].shape[0]):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0][j, :, :, :, i]),
                origin=origin_left, spacing=spacing, direction=direction)
            if i > 0:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if j == 1:  # flipped
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                    origin=probability_image.origin, spacing=probability_image.spacing,
                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                probability_images_left.append(probability_image)
            else:       # flipped
                probability_images_right.append(probability_image)


    ################################
    #
    # Left:  do prediction of mtl, hippocampal, and ec regions and normalize to native space
    #
    ################################

    for i in range(1, len(predicted_data)):
        for j in range(predicted_data[i].shape[0]):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[i][j, :, :, :, 0]),
                origin=origin_left, spacing=spacing, direction=direction)
            probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)

            if j == 1:  # flipped
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                    origin=probability_image.origin, spacing=probability_image.spacing,
                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                foreground_probability_images_left.append(probability_image)
            else:
                foreground_probability_images_right.append(probability_image)

    ################################
    #
    # Right:  build model and load weights
    #
    ################################

    network_name = 'deepFlashRightT1'
    if t2 is not None:
        network_name = 'deepFlashRightBoth'

    if use_hierarchical_parcellation:
        network_name += "Hierarchical"

    if use_rank_intensity:
        network_name += "_ri"

    if verbose:
        print("DeepFlash: retrieving model weights (right).")
    weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Right:  do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Prediction (right).")

    batchX = None
    if use_contralaterality:
        batchX = np.zeros((2, *image_size, channel_size))
    else:
        batchX = np.zeros((1, *image_size, channel_size))

    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    if use_rank_intensity:
        t1_cropped = ants.rank_intensity(t1_cropped)
    else:
        t1_cropped = ants.histogram_match_image(t1_cropped, t1_template_roi_right, 255, 64, False)
    batchX[0,:,:,:,0] = t1_cropped.numpy()
    if use_contralaterality:
        t1_cropped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_right, upper_bound_right)
        if use_rank_intensity:
            t1_cropped = ants.rank_intensity(t1_cropped)
        else:
            t1_cropped = ants.histogram_match_image(t1_cropped, t1_template_roi_right, 255, 64, False)
        batchX[1,:,:,:,0] = t1_cropped.numpy()
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_right, upper_bound_right)
        if use_rank_intensity:
            t2_cropped = ants.rank_intensity(t2_cropped)
        else:
            t2_cropped = ants.histogram_match_image(t2_cropped, t2_template_roi_right, 255, 64, False)
        batchX[0,:,:,:,1] = t2_cropped.numpy()
        if use_contralaterality:
            t2_cropped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_right, upper_bound_right)
            if use_rank_intensity:
                t2_cropped = ants.rank_intensity(t2_cropped)
            else:
                t2_cropped = ants.histogram_match_image(t2_cropped, t2_template_roi_right, 255, 64, False)
            batchX[1,:,:,:,1] = t2_cropped.numpy()

    for i in range(len(priors_image_right_list)):
        cropped_prior = ants.crop_indices(priors_image_right_list[i], lower_bound_right, upper_bound_right)
        for j in range(batchX.shape[0]):
            batchX[j,:,:,:,i + (channel_size - len(labels_right))] = cropped_prior.numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    for i in range(1 + len(labels_right)):
        for j in range(predicted_data[0].shape[0]):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0][j, :, :, :, i]),
                origin=origin_right, spacing=spacing, direction=direction)
            if i > 0:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if j == 1:  # flipped
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                    origin=probability_image.origin, spacing=probability_image.spacing,
                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                if use_contralaterality:
                    probability_images_right[i] = (probability_images_right[i] + probability_image) / 2
                else:
                    probability_images_right.append(probability_image)
            else:       # flipped
                probability_images_left[i] = (probability_images_left[i] + probability_image) / 2


    ################################
    #
    # Right:  do prediction of mtl, hippocampal, and ec regions and normalize to native space
    #
    ################################

    for i in range(1, len(predicted_data)):
        for j in range(predicted_data[i].shape[0]):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[i][j, :, :, :, 0]),
                origin=origin_right, spacing=spacing, direction=direction)
            probability_image = ants.decrop_image(probability_image, t1_preprocessed * 0)

            if j == 1:  # flipped
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image = ants.from_numpy(probability_array_flipped,
                    origin=probability_image.origin, spacing=probability_image.spacing,
                    direction=probability_image.direction)

            if do_preprocessing:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)

            if j == 0:  # not flipped
                if use_contralaterality:
                    foreground_probability_images_right[i-1] = (foreground_probability_images_right[i-1] + probability_image) / 2
                else:
                    foreground_probability_images_right.append(probability_image)
            else:
                foreground_probability_images_left[i-1] = (foreground_probability_images_left[i-1] + probability_image) / 2

    ################################
    #
    # Combine priors
    #
    ################################

    probability_background_image = ants.image_clone(t1) * 0
    for i in range(1, len(probability_images_left)):
        probability_background_image += probability_images_left[i]
    for i in range(1, len(probability_images_right)):
        probability_background_image += probability_images_right[i]

    probability_images.append(probability_background_image * -1 + 1)
    for i in range(1, len(probability_images_left)):
        probability_images.append(probability_images_left[i])
        probability_images.append(probability_images_right[i])

    ################################
    #
    # Convert probability images to segmentation
    #
    ################################

    # image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    # segmentation_matrix = np.argmax(image_matrix, axis=0)
    # segmentation_image = ants.matrix_to_images(
    #     np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    foreground_probability_images = list()
    for i in range(len(foreground_probability_images_left)):
        foreground_probability_images.append(foreground_probability_images_left[i] + foreground_probability_images_right[i])

    return_dict = None
    if use_hierarchical_parcellation:
        return_dict = {'segmentation_image' : relabeled_image,
                       'probability_images' : probability_images,
                       'medial_temporal_lobe_probability_image' : foreground_probability_images[0],
                       'other_region_probability_image' : foreground_probability_images[1],
                       'hippocampal_probability_image' : foreground_probability_images[2]
                      }
    else:
        return_dict = {'segmentation_image' : relabeled_image,
                       'probability_images' : probability_images,
                       'medial_temporal_lobe_probability_image' : foreground_probability_images[0]
                      }

    return(return_dict)


def deep_flash_deprecated(t1,
               do_preprocessing=True,
               do_per_hemisphere=True,
               which_hemisphere_models="new",
               antsxnet_cache_directory=None,
               verbose=False
               ):

    """
    Hippocampal/Enthorhinal segmentation using "Deep Flash"

    Perform hippocampal/entorhinal segmentation in T1 images using
    labels from Mike Yassa's lab

    https://faculty.sites.uci.edu/myassa/

    The labeling is as follows:
    Label 0 :  background
    Label 5 :  left aLEC
    Label 6 :  right aLEC
    Label 7 :  left pMEC
    Label 8 :  right pMEC
    Label 9 :  left perirhinal
    Label 10:  right perirhinal
    Label 11:  left parahippocampal
    Label 12:  right parahippocampal
    Label 13:  left DG/CA3
    Label 14:  right DG/CA3
    Label 15:  left CA1
    Label 16:  right CA1
    Label 17:  left subiculum
    Label 18:  right subiculum

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * denoising,
       * brain extraction, and
       * affine registration to MNI.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    do_preprocessing : boolean
        See description above.

    do_per_hemisphere : boolean
        If True, do prediction based on separate networks per hemisphere.  Otherwise,
        use the single network trained for both hemispheres.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_flash(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size

    print("This function is deprecated.  Please update to deep_flash().")

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = t1
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            template="croppedMni152",
            template_transform_type="antsRegistrationSyNQuickRepro[a]",
            do_bias_correction=True,
            do_denoising=True,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

    probability_images = list()
    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    ################################
    #
    # Process left/right in same network
    #
    ################################

    if do_per_hemisphere == False:

        ################################
        #
        # Build model and load weights
        #
        ################################

        template_size = (160, 192, 160)

        unet_model = create_unet_model_3d((*template_size, 1),
            number_of_outputs=len(labels),
            number_of_layers=4, number_of_filters_at_base_layer=8, dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            weight_decay=1e-5, additional_options=("attentionGating",))

        if verbose:
            print("DeepFlash: retrieving model weights.")

        weights_file_name = get_pretrained_network("deepFlash", antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction.")

        cropped_image = pad_or_crop_image_to_size(t1_preprocessed, template_size)

        batchX = np.expand_dims(cropped_image.numpy(), axis=0)
        batchX = np.expand_dims(batchX, axis=-1)
        batchX = (batchX - batchX.mean()) / batchX.std()

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        origin = cropped_image.origin
        spacing = cropped_image.spacing
        direction = cropped_image.direction

        for i in range(len(labels)):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
                origin=origin, spacing=spacing, direction=direction)
            if i > 0:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if do_preprocessing:
                probability_images.append(ants.apply_transforms(fixed=t1,
                    moving=decropped_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose))
            else:
                probability_images.append(decropped_image)

    ################################
    #
    # Process left/right in split networks
    #
    ################################

    else:

        ################################
        #
        # Left:  download spatial priors
        #
        ################################

        spatial_priors_left_file_name_path = get_antsxnet_data("priorDeepFlashLeftLabels",
          antsxnet_cache_directory=antsxnet_cache_directory)
        spatial_priors_left = ants.image_read(spatial_priors_left_file_name_path)
        priors_image_left_list = ants.ndimage_to_list(spatial_priors_left)

        ################################
        #
        # Left:  build model and load weights
        #
        ################################

        template_size = (64, 96, 96)
        labels_left = (0, 5, 7, 9, 11, 13, 15, 17)
        channel_size = 1 + len(labels_left)

        number_of_filters = 16
        network_name = ''
        if which_hemisphere_models == "old":
           network_name = "deepFlashLeft16"
        elif which_hemisphere_models == "new":
           network_name = "deepFlashLeft16new"
        else:
            raise ValueError("network_name must be \"old\" or \"new\".")

        unet_model = create_unet_model_3d((*template_size, channel_size),
            number_of_outputs = len(labels_left),
            number_of_layers = 4, number_of_filters_at_base_layer = number_of_filters, dropout_rate = 0.0,
            convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
            weight_decay = 1e-5, additional_options=("attentionGating",))

        if verbose:
            print("DeepFlash: retrieving model weights (left).")
        weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Left:  do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction (left).")

        cropped_image = ants.crop_indices(t1_preprocessed, (30, 51, 0), (94, 147, 96))
        image_array = cropped_image.numpy()
        image_array = (image_array - image_array.mean()) / image_array.std()

        batchX = np.zeros((1, *template_size, channel_size))
        batchX[0,:,:,:,0] = image_array

        for i in range(len(priors_image_left_list)):
            cropped_prior = ants.crop_indices(priors_image_left_list[i], (30, 51, 0), (94, 147, 96))
            batchX[0,:,:,:,i+1] = cropped_prior.numpy()

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        origin = cropped_image.origin
        spacing = cropped_image.spacing
        direction = cropped_image.direction

        probability_images_left = list()
        for i in range(len(labels_left)):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
                origin=origin, spacing=spacing, direction=direction)
            if i > 0:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if do_preprocessing:
                probability_images_left.append(ants.apply_transforms(fixed=t1,
                    moving=decropped_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose))
            else:
                probability_images_left.append(decropped_image)

        ################################
        #
        # Right:  download spatial priors
        #
        ################################

        spatial_priors_right_file_name_path = get_antsxnet_data("priorDeepFlashRightLabels",
          antsxnet_cache_directory=antsxnet_cache_directory)
        spatial_priors_right = ants.image_read(spatial_priors_right_file_name_path)
        priors_image_right_list = ants.ndimage_to_list(spatial_priors_right)

        ################################
        #
        # Right:  build model and load weights
        #
        ################################

        template_size = (64, 96, 96)
        labels_right = (0, 6, 8, 10, 12, 14, 16, 18)
        channel_size = 1 + len(labels_right)

        number_of_filters = 16
        network_name = ''
        if which_hemisphere_models == "old":
           network_name = "deepFlashRight16"
        elif which_hemisphere_models == "new":
           network_name = "deepFlashRight16new"
        else:
            raise ValueError("network_name must be \"old\" or \"new\".")

        unet_model = create_unet_model_3d((*template_size, channel_size),
            number_of_outputs = len(labels_right),
            number_of_layers = 4, number_of_filters_at_base_layer = number_of_filters, dropout_rate = 0.0,
            convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
            weight_decay = 1e-5, additional_options=("attentionGating",))

        weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Right:  do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction (right).")

        cropped_image = ants.crop_indices(t1_preprocessed, (88, 51, 0), (152, 147, 96))
        image_array = cropped_image.numpy()
        image_array = (image_array - image_array.mean()) / image_array.std()

        batchX = np.zeros((1, *template_size, channel_size))
        batchX[0,:,:,:,0] = image_array

        for i in range(len(priors_image_right_list)):
            cropped_prior = ants.crop_indices(priors_image_right_list[i], (88, 51, 0), (152, 147, 96))
            batchX[0,:,:,:,i+1] = cropped_prior.numpy()

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        origin = cropped_image.origin
        spacing = cropped_image.spacing
        direction = cropped_image.direction

        probability_images_right = list()
        for i in range(len(labels_right)):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
                origin=origin, spacing=spacing, direction=direction)
            if i > 0:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if do_preprocessing:
                probability_images_right.append(ants.apply_transforms(fixed=t1,
                    moving=decropped_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose))
            else:
                probability_images_right.append(decropped_image)

        ################################
        #
        # Combine priors
        #
        ################################

        probability_background_image = ants.image_clone(t1) * 0
        for i in range(1, len(probability_images_left)):
            probability_background_image += probability_images_left[i]
        for i in range(1, len(probability_images_right)):
            probability_background_image += probability_images_right[i]

        probability_images.append(probability_background_image * -1 + 1)
        for i in range(1, len(probability_images_left)):
            probability_images.append(probability_images_left[i])
            probability_images.append(probability_images_right[i])

    ################################
    #
    # Convert probability images to segmentation
    #
    ################################

    # image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    # segmentation_matrix = np.argmax(image_matrix, axis=0)
    # segmentation_image = ants.matrix_to_images(
    #     np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    return_dict = {'segmentation_image' : relabeled_image,
                   'probability_images' : probability_images}
    return(return_dict)


