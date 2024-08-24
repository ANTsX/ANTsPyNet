import ants
import numpy as np
import tensorflow as tf

def lesion_segmentation(t1,
                        do_preprocessing=True,
                        verbose=False):

    """
    Lesion segmentation from T1-w images.

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    do_preprocessing : boolean
        perform n4 bias correction, intensity truncation, brain extraction.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    lesion segmentation probability image

    Example
    -------
    >>> t1 = ants.image_read("t1.nii.gz")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import brain_extraction

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = None
    brain_mask = None

    if do_preprocessing:
        if verbose:
            print("Preprocess T1 images.")

        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=False,
            verbose=verbose)
        brain_mask = t1_preprocessing["brain_mask"]
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask
    else:
        t1_preprocessed = ants.image_clone(t1)
        brain_mask = ants.threshold_image(t1_preprocessed, 0, 0, 0, 1)

    template_size = (192, 208, 192)
    template = ants.image_read(get_antsxnet_data('mni152'))
    template = pad_or_crop_image_to_size(template, template_size)
    template_mask = brain_extraction(template, modality="t1", verbose=verbose)
    template = template * template_mask

    if verbose:
        print("Load u-net models and weights.")

    number_of_classification_labels = 1
    channel_size = 1
    unet_weights_file_name = get_pretrained_network("lesion_whole_brain")

    unet_model = create_unet_model_3d((*template_size, channel_size),
        number_of_outputs=number_of_classification_labels,
        mode='sigmoid',
        number_of_filters=(16, 32, 64, 128, 256), dropout_rate=0.0,
        convolution_kernel_size=3, deconvolution_kernel_size=2,
        weight_decay=1e-5, additional_options=("attentionGating",))
    unet_model.load_weights(unet_weights_file_name)

    if verbose:
        print("Alignment to template.")

    image_min = t1_preprocessed[brain_mask != 0].min()
    image_max = t1_preprocessed[brain_mask != 0].max()

    registration = ants.registration(template, t1_preprocessed, type_of_transform="antsRegistrationSyNQuick[a]",
                                     verbose=verbose)
    image = registration['warpedmovout']
    image = (image - image_min) / (image_max - image_min)

    batchX = np.zeros((1, *image.shape, channel_size))
    batchX[0,:,:,:,0] = image.numpy()

    lesion_mask_array = np.squeeze(unet_model.predict(batchX, verbose=verbose))
    lesion_mask = ants.copy_image_info(template, ants.from_numpy(lesion_mask_array))

    probability_image = ants.apply_transforms(t1_preprocessed, lesion_mask, registration['invtransforms'],
                                              whichtoinvert=[True], verbose=verbose)

    return(probability_image)


def lesion_segmentation_experimental(t1,
                                     which_model=0,
                                     prediction_batch_size=16,
                                     patch_stride_length=32,
                                     do_preprocessing=True,
                                     verbose=False):

    """
    Emily lesion segmentation.

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    which_model : int
        type of prediction (otherwise do whole-brain prediction).

    prediction_batch_size : int
        Control memory usage for prediction.  More consequential for GPU-usage.

    patch_stride_length : 3-D tuple or int
        Dictates the stride length for accumulating predicting patches.

    do_preprocessing : boolean
        perform n4 bias correction, intensity truncation, brain extraction.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> t1 = ants.image_read("t1.nii.gz")
    """

    from ..architectures import create_sysu_media_unet_model_3d
    from ..architectures import create_unet_model_3d
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import brain_extraction

    if which_model == 0 and np.any(t1.shape < np.array((64, 64, 64))):
        raise ValueError("Images must be > 64 voxels per dimension.")

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = None
    brain_mask = None

    if do_preprocessing:
        if verbose:
            print("Preprocess T1 images.")

        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1",
            do_bias_correction=True,
            do_denoising=False,
            verbose=verbose)
        brain_mask = t1_preprocessing["brain_mask"]
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask
    else:
        t1_preprocessed = ants.image_clone(t1)
        brain_mask = ants.threshold_image(t1_preprocessed, 0, 0, 0, 1)

    if which_model == 0:

        ################################
        #
        # Build model and load weights
        #
        ################################

        if verbose:
            print("Load u-net models and weights.")

        patch_size = (64, 64, 64)
        if isinstance(patch_stride_length, int):
            patch_stride_length = (patch_stride_length,) * 3
        channel_size = 1

        unet_model = create_sysu_media_unet_model_3d((*patch_size, channel_size),
            number_of_filters=(64, 96, 128, 256, 512))

        unet_weights_file_name = get_pretrained_network("lesion_patch")
        unet_model.load_weights(unet_weights_file_name)

        ################################
        #
        # Extract patches
        #
        ################################

        if verbose:
            print("Extract patches.")

        image = t1_preprocessed
        image = (image - image.min()) / (image.max() - image.min())

        image_patches = extract_image_patches(image,
                                        patch_size=patch_size,
                                        max_number_of_patches="all",
                                        stride_length=patch_stride_length,
                                        mask_image=brain_mask,
                                        random_seed=None,
                                        return_as_array=True)
        total_number_of_patches = image_patches.shape[0]

        ################################
        #
        # Do prediction and then restack into the image
        #
        ################################

        number_of_batches = total_number_of_patches // prediction_batch_size
        residual_number_of_patches = total_number_of_patches - number_of_batches * prediction_batch_size
        if residual_number_of_patches > 0:
            number_of_batches = number_of_batches + 1

        if verbose:
            print("  Total number of patches: ", str(total_number_of_patches))
            print("  Prediction batch size: ", str(prediction_batch_size))
            print("  Number of batches: ", str(number_of_batches))

        unet_prediction = np.zeros((total_number_of_patches, *patch_size, channel_size))
        for b in range(number_of_batches):
            batchX = None
            if b < number_of_batches - 1 or residual_number_of_patches == 0:
                batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
            else:
                batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

            indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
            batchX[:,:,:,:,0] = image_patches[indices,:,:,:]

            if verbose:
                print("Predicting unet batch ", str(b + 1), " of ", str(number_of_batches))
            unet_prediction[indices,:,:,:,:] = unet_model.predict(batchX, verbose=verbose)

        if verbose:
            print("Predict patches and reconstruct.")

        probability_image = reconstruct_image_from_patches(np.squeeze(unet_prediction[:,:,:,:,0]),
                                                            stride_length=patch_stride_length,
                                                            domain_image=brain_mask,
                                                            domain_image_is_mask=True)
        return(probability_image)

    elif which_model == 1:

        template_size = (192, 208, 192)
        template = ants.image_read(get_antsxnet_data('mni152'))
        template = pad_or_crop_image_to_size(template, template_size)
        template_mask = brain_extraction(template, modality="t1", verbose=True)
        template = template * template_mask

        if verbose:
            print("Load u-net models and weights.")

        number_of_classification_labels = 1
        channel_size = 1
        unet_weights_file_name = get_pretrained_network("lesion_whole_brain")

        unet_model = create_unet_model_3d((*template_size, channel_size),
            number_of_outputs=number_of_classification_labels,
            mode='sigmoid',
            number_of_filters=(16, 32, 64, 128, 256), dropout_rate=0.0,
            convolution_kernel_size=3, deconvolution_kernel_size=2,
            weight_decay=1e-5, additional_options=("attentionGating",))
        unet_model.load_weights(unet_weights_file_name)

        if verbose:
            print("Alignment to template.")

        registration = ants.registration(template, t1_preprocessed, type_of_transform="antsRegistrationSyNQuick[a]",
                                         verbose=verbose)
        image = registration['warpedmovout']
        image = (image - image.min()) / (image.max() - image.min())

        batchX = np.zeros((1, *image.shape, channel_size))
        batchX[0,:,:,:,0] = image.numpy()

        lesion_mask_array = np.squeeze(unet_model.predict(batchX, verbose=verbose))
        lesion_mask = ants.copy_image_info(template, ants.from_numpy(lesion_mask_array))

        probability_image = ants.apply_transforms(t1_preprocessed, lesion_mask, registration['invtransforms'],
                                                  whichtoinvert=[True], verbose=verbose)

        return(probability_image)

    elif which_model == 2:

        template_size = (192, 208, 192)
        template = ants.image_read(get_antsxnet_data('mni152'))
        template = pad_or_crop_image_to_size(template, template_size)
        template_mask = brain_extraction(template, modality="t1", verbose=True)
        template = template * template_mask

        if verbose:
            print("Load u-net models and weights.")

        number_of_classification_labels = 1
        channel_size = 2
        unet_weights_file_name = get_pretrained_network("lesion_flip_brain")

        unet_model = create_unet_model_3d((*template_size, channel_size),
            number_of_outputs=number_of_classification_labels,
            mode='sigmoid',
            number_of_filters=(16, 32, 64, 128, 256), dropout_rate=0.0,
            convolution_kernel_size=3, deconvolution_kernel_size=2,
            weight_decay=1e-5, additional_options=("attentionGating",))
        unet_model.load_weights(unet_weights_file_name)

        if verbose:
            print("Alignment to template.")

        registration = ants.registration(template, t1_preprocessed, type_of_transform="antsRegistrationSyNQuick[a]",
                                         verbose=verbose)
        image = registration['warpedmovout']
        image = (image - image.min()) / (image.max() - image.min())

        batchX = np.zeros((1, *image.shape, channel_size))
        batchX[0,:,:,:,0] = image.numpy()
        batchX[0,:,:,:,1] = np.flip(batchX[0,:,:,:,0], axis=0)

        lesion_mask_array = np.squeeze(unet_model.predict(batchX, verbose=verbose))
        lesion_mask = ants.copy_image_info(template, ants.from_numpy(lesion_mask_array))

        probability_image = ants.apply_transforms(t1_preprocessed, lesion_mask, registration['invtransforms'],
                                                  whichtoinvert=[True], verbose=verbose)

        return(probability_image)

    elif which_model == 3:

        template_size = (192, 208, 192)
        template = ants.image_read(get_antsxnet_data('mni152'))
        template = pad_or_crop_image_to_size(template, template_size)
        template_mask = brain_extraction(template, modality="t1", verbose=True)
        template = template * template_mask

        if verbose:
            print("Load u-net models and weights.")

        number_of_classification_labels = 1
        channel_size = 3
        unet_weights_file_name = get_pretrained_network("lesion_flip_template_brain")

        unet_model = create_unet_model_3d((*template_size, channel_size),
            number_of_outputs=number_of_classification_labels,
            mode='sigmoid',
            number_of_filters=(16, 32, 64, 128, 256), dropout_rate=0.0,
            convolution_kernel_size=3, deconvolution_kernel_size=2,
            weight_decay=1e-5, additional_options=("attentionGating",))
        unet_model.load_weights(unet_weights_file_name)

        if verbose:
            print("Alignment to template.")

        registration = ants.registration(template, t1_preprocessed, type_of_transform="antsRegistrationSyNQuick[a]",
                                         verbose=verbose)
        image = registration['warpedmovout']
        image = (image - image.min()) / (image.max() - image.min())

        batchX = np.zeros((1, *image.shape, channel_size))
        batchX[0,:,:,:,0] = image.numpy()
        batchX[0,:,:,:,1] = np.flip(batchX[0,:,:,:,0], axis=0)
        batchX[0,:,:,:,2] = (ants.histogram_match_image(template, image)).numpy()

        lesion_mask_array = np.squeeze(unet_model.predict(batchX, verbose=verbose))
        lesion_mask = ants.copy_image_info(template, ants.from_numpy(lesion_mask_array))

        probability_image = ants.apply_transforms(t1_preprocessed, lesion_mask, registration['invtransforms'],
                                                  whichtoinvert=[True], verbose=verbose)

        return(probability_image)

