import ants
import numpy as np
import tensorflow as tf
from tensorflow import keras


def brain_tumor_segmentation(flair,
                     t1,
                     t1_contrast,
                     t2,
                     prediction_batch_size=16,
                     patch_stride_length=32,
                     do_preprocessing=True,
                     verbose=False):

    """
    Perform brain tumor probabilistic segmentation given pre-aligned
    FLAIR, T1, T1 contrast, and T2 images.  Note that the underlying
    model is 3-D and requires images to be of > 64 voxels in each
    dimension.

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * brain extraction

    All input images should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    \code{do_preprocessing = True}

    Arguments
    ---------
    flair : ANTsImage
        input 3-D FLAIR brain image (not skull-stripped).

    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    t1_contrast : ANTsImage
        input 3-D T1 contrast brain image (not skull-stripped).

    t2 : ANTsImage
        input 3-D T2 brain image (not skull-stripped).

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
    >>> flair = ants.image_read("flair.nii.gz")
    >>> t1 = ants.image_read("t1.nii.gz")
    """

    from ..architectures import create_sysu_media_unet_model_3d
    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image

    if np.any(t1.shape < np.array((64, 64, 64))):
        raise ValueError("Images must be > 64 voxels per dimension.")

    ################################
    #
    # Preprocess images
    #
    ################################

    flair_preprocessed = None
    t1_preprocessed = None
    t1_contrast_preprocessed = None
    t2_preprocessed = None
    brain_mask = None

    if do_preprocessing:
        if verbose:
            print("Preprocess FLAIR, T1, T1 contrast, and T2 images.")

        do_bias_correction = False

        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1",
            do_bias_correction=do_bias_correction,
            do_denoising=False,
            verbose=verbose)
        brain_mask = ants.threshold_image(t1_preprocessing["brain_mask"], 0.5, 1, 1, 0)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask

        flair_preprocessing = preprocess_brain_image(flair,
            truncate_intensity=None,
            brain_extraction_modality=None,
            do_bias_correction=do_bias_correction,
            do_denoising=False,
            verbose=verbose)
        flair_preprocessed = flair_preprocessing["preprocessed_image"] * brain_mask

        t1_contrast_preprocessing = preprocess_brain_image(t1_contrast,
            truncate_intensity=None,
            brain_extraction_modality=None,
            do_bias_correction=do_bias_correction,
            do_denoising=False,
            verbose=verbose)
        t1_contrast_preprocessed = t1_contrast_preprocessing["preprocessed_image"] * brain_mask

        t2_preprocessing = preprocess_brain_image(t2,
            truncate_intensity=None,
            brain_extraction_modality=None,
            do_bias_correction=do_bias_correction,
            do_denoising=False,
            verbose=verbose)
        t2_preprocessed = t2_preprocessing["preprocessed_image"] * brain_mask

    else:
        flair_preprocessed = ants.image_clone(flair)
        t1_preprocessed = ants.image_clone(t1)
        t1_contrast_preprocessed = ants.image_clone(t1_contrast)
        t2_preprocessed = ants.image_clone(t2)
        brain_mask = ants.threshold_image(flair_preprocessed, 0, 0, 0, 1)

    images = [flair_preprocessed, t1_preprocessed, t1_contrast_preprocessed, t2_preprocessed]

    indices = brain_mask > 0
    for i in range(len(images)):
        images[i] = (images[i] - images[i][indices].min()) / (images[i][indices].max() - images[i][indices].min())


    ################################################################################################
    #
    #                        Stage 1:  Whole tumor segmentation
    #
    ################################################################################################

    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Stage 1:  Load model and weights.")

    patch_size = (64, 64, 64)
    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3
    number_of_filters = (64, 96, 128, 256, 512)
    channel_size = 4

    model = create_sysu_media_unet_model_3d((*patch_size, channel_size),
                                             number_of_filters=number_of_filters)
    weights_file_name = get_pretrained_network("bratsStage1")
    model.load_weights(weights_file_name)

    ################################
    #
    # Extract patches
    #
    ################################

    if verbose:
        print("Stage 1:  Extract patches.")

    image_patches = list()
    for i in range(len(images)):
        image_patches.append(ants.extract_image_patches(images[i],
                                           patch_size=patch_size,
                                           max_number_of_patches="all",
                                           stride_length=patch_stride_length,
                                           mask_image=brain_mask,
                                           random_seed=None,
                                           return_as_array=True))
    total_number_of_patches = image_patches[0].shape[0]

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

    prediction = np.zeros((total_number_of_patches, *patch_size, 1))
    for b in range(number_of_batches):
        batchX = None
        if b < number_of_batches - 1 or residual_number_of_patches == 0:
            batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
        else:
            batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

        indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
        for i in range(len(images)):
            batchX[:,:,:,:,i] = image_patches[i][indices,:,:,:]

        if verbose:
            print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))
        prediction[indices,:,:,:,:] = model.predict(batchX, verbose=verbose)

    if verbose:
        print("Stage 1:  Predict patches and reconstruct.")

    tumor_probability_image = ants.reconstruct_image_from_patches(np.squeeze(prediction),
                                                           stride_length=patch_stride_length,
                                                           domain_image=brain_mask,
                                                           domain_image_is_mask=True)

    tumor_mask = ants.threshold_image(tumor_probability_image, 0.5, 1.0, 1, 0)

    ################################################################################################
    #
    #                        Stage 2:  Tumor component segmentation
    #
    ################################################################################################

    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Stage 2:  Load model and weights.")

    patch_size = (64, 64, 64)
    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3
    channel_size = 5  # [FLAIR, T1, T1GD, T2, MASK]
    number_of_classification_labels = 5

    model = create_unet_model_3d((*patch_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="sigmoid",
        number_of_filters=(32, 64, 128, 256, 512),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0, weight_decay=0)

    weights_file_name = get_pretrained_network("bratsStage2")
    model.load_weights(weights_file_name)

    ################################
    #
    # Extract patches
    #
    ################################

    if verbose:
        print("Stage 2:  Extract patches.")

    images.append(tumor_mask)

    image_patches = list()
    for i in range(len(images)):
        image_patches.append(ants.extract_image_patches(images[i],
                                           patch_size=patch_size,
                                           max_number_of_patches="all",
                                           stride_length=patch_stride_length,
                                           mask_image=tumor_mask,
                                           random_seed=None,
                                           return_as_array=True))
    total_number_of_patches = image_patches[0].shape[0]

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

    prediction = np.zeros((total_number_of_patches, *patch_size, channel_size))
    for b in range(number_of_batches):
        batchX = None
        if b < number_of_batches - 1 or residual_number_of_patches == 0:
            batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
        else:
            batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

        indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
        for i in range(len(images)):
            batchX[:,:,:,:,i] = image_patches[i][indices,:,:,:]

        if verbose:
            print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))
        prediction[indices,:,:,:,:] = model.predict(batchX, verbose=verbose)

    if verbose:
        print("Stage 2:  Predict patches and reconstruct.")

    probability_images = list()
    for c in range(number_of_classification_labels):
        probability_images.append(ants.reconstruct_image_from_patches(np.squeeze(prediction[:,:,:,:,c]),
                                                           stride_length=patch_stride_length,
                                                           domain_image=tumor_mask,
                                                           domain_image_is_mask=True))

    image_matrix = ants.image_list_to_matrix(probability_images, tumor_mask * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return(return_dict)

