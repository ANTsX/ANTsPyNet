import os

import numpy as np
import keras

import ants

def deep_atropos(t1,
                 do_preprocessing=True,
                 output_directory=None,
                 verbose=False):

    """
    Six-tissue segmentation.

    Perform Atropos-style six tissue segmentation using deep learning.

    The labeling is as follows:
    Label 0 :  background
    Label 1 :  CSF
    Label 2 :  gray matter
    Label 3 :  white matter
    Label 4 :  deep gray matter
    Label 5 :  brain stem
    Label 6 :  cerebellum

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

    output_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        tempfile.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_atropos(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import categorical_focal_loss
    from ..utilities import preprocess_brain_image
    from ..utilities import crop_image_center

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = t1
    if do_preprocessing == True:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            do_brain_extraction=True,
            template="croppedMni152",
            template_transform_type="AffineFast",
            do_bias_correction=True,
            do_denoising=True,
            output_directory=output_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

    ################################
    #
    # Build model and load weights
    #
    ################################

    template_size = (160, 192, 160)
    labels = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

    unet_model = create_unet_model_3d((*template_size, 1),
        number_of_outputs = len(labels),
        number_of_layers = 4, number_of_filters_at_base_layer = 8, dropout_rate = 0.0,
        convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
        weight_decay = 1e-5, add_attention_gating=True)

    weights_file_name = None
    if output_directory is not None:
        weights_file_name = output_directory + "/sixTissueWeights.h5"
        if not os.path.exists(weights_file_name):
            if verbose == True:
                print("Deep Atropos:  downloading model weights.")
            weights_file_name = get_pretrained_network("sixTissueBrainSegmentation", weights_file_name)
    else:
        weights_file_name = get_pretrained_network("sixTissueBrainSegmentation")

    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose == True:
        print("Prediction.")

    cropped_image = ants.crop_indices(t1_preprocessed, (12, 14, 0), (171, 205, 159))

    batchX = np.expand_dims(cropped_image.numpy(), axis=0)
    batchX = np.expand_dims(batchX, axis=-1)
    batchX = (batchX - batchX.mean()) / batchX.std()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    origin = cropped_image.origin
    spacing = cropped_image.spacing
    direction = cropped_image.direction

    probability_images = list()
    for i in range(len(labels)):
        probability_image = \
            ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
            origin=origin, spacing=spacing, direction=direction)
        if i > 0:
            decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
        else:
            decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

        if do_preprocessing == True:
            probability_images.append(ants.apply_transforms(fixed=t1,
                moving=decropped_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", verbose=verbose))
        else:
            probability_images.append(decropped_image)

    image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    return_dict = {'segmentation_image'  : relabeled_image,
                   'probability_images' : probability_images}
    return(return_dict)
