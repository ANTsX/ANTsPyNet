import numpy as np
import ants

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def deep_flash(t1,
                t2=None,
                do_preprocessing=True,
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
    each label and foreground.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_flash(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image

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
    t1_preprocessing = None
    t1_preprocessed_flipped = None
    if do_preprocessing == True:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.995),
            brain_extraction_modality="t1",
            template="deepFlashTemplateT1",
            template_transform_type="antsRegistrationSyNQuick[a]",
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"]
        t1_preprocessed = (t1_preprocessed - t1_preprocessed.mean()) / t1_preprocessed.std()
        t1_preprocessed_array = t1_preprocessed.numpy()        
        t1_preprocessed_array_flipped = np.flip(t1_preprocessed_array, axis=0)
        t1_preprocessed_flipped = ants.from_numpy(t1_preprocessed_array_flipped, 
                                                  origin=t1_preprocessed.origin,
                                                  spacing=t1_preprocessed.spacing,
                                                  direction=t1_preprocessed.direction)

    t2_preprocessed = t2
    if t2 is not None:
        t2_preprocessing = preprocess_brain_image(t2,
            truncate_intensity=(0.01, 0.995),
            brain_extraction_modality=None,
            template_transform_type=None,
            do_bias_correction=True,
            do_denoising=False,
            antsxnet_cache_directory=antsxnet_cache_directory,
            verbose=verbose)
        t2_preprocessed = ants.apply_transforms(fixed=t1_preprocessed,
            moving=t2_preprocessing["preprocessed_image"],
            transformlist=t1_preprocessing['template_transforms']['fwdtransforms'],
            verbose=verbose)
        t2_preprocessed = (t2_preprocessed - t2_preprocessed.mean()) / t2_preprocessed.std()
        t2_preprocessed_array = t1_preprocessed.numpy()        
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

    labels_left = labels[1::2]
    priors_image_left_list = priors_image_list[1::2]
    probability_images_left = list()
    foreground_probability_image_left = None
    lower_bound_left = (76, 74, 56)
    upper_bound_left = (140, 138, 152)
    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    origin_left = tmp_cropped.origin

    spacing = tmp_cropped.spacing
    direction = tmp_cropped.direction

    labels_right = labels[2::2]
    priors_image_right_list = priors_image_list[2::2]
    probability_images_right = list()
    foreground_probability_image_right = None
    lower_bound_right = (20, 74, 56)
    upper_bound_right = (84, 138, 152)
    tmp_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    origin_right = tmp_cropped.origin

    ################################
    #
    # Left:  build model and load weights
    #
    ################################

    channel_size = 1 + len(labels_left)
    number_of_classification_labels = 1 + len(labels_left)

    network_name = 'deepFlashLeftT1'
    if t2 is not None:
        network_name = 'deepFlashLeftBoth'
        channel_size += 1

    unet_model = create_unet_model_3d((*image_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0, weight_decay=0)

    penultimate_layer = unet_model.layers[-2].output
    output = Conv3D(filters=1,
                    kernel_size=(1, 1, 1),
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)
    unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output])

    if verbose == True:
        print("DeepFlash: retrieving model weights (left).")
    weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Left:  do prediction and normalize to native space
    #
    ################################

    if verbose == True:
        print("Prediction (left).")

    batchX = np.zeros((2, *image_size, channel_size))

    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_left, upper_bound_left)
    ants.image_write(t1_cropped, "~/Desktop/t1_cropped.nii.gz")
    batchX[0,:,:,:,0] = t1_cropped.numpy()
    t1_cropped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_left, upper_bound_left)
    ants.image_write(t1_cropped, "~/Desktop/t1_cropped_flipped.nii.gz")
    batchX[1,:,:,:,0] = t1_cropped.numpy()
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_left, upper_bound_left)
        batchX[0,:,:,:,1] = t2_cropped.numpy()
        t2_cropped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_left, upper_bound_left)
        batchX[1,:,:,:,1] = t2_cropped.numpy()

    for i in range(len(priors_image_left_list)):
        ants.image_write(priors_image_left_list[i], "~/Desktop/prior" + str(i) + ".nii.gz")
        cropped_prior = ants.crop_indices(priors_image_left_list[i], lower_bound_left, upper_bound_left)
        ants.image_write(cropped_prior, "~/Desktop/prior_cropped" + str(i) + ".nii.gz")
        batchX[0,:,:,:,i + (channel_size - len(labels_left))] = cropped_prior.numpy()
        batchX[1,:,:,:,i + (channel_size - len(labels_left))] = cropped_prior.numpy()
    raise ValueError("Test")    

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    for i in range(1 + len(labels_left)):
        for j in range(2):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0][j, :, :, :, i]),
                origin=origin_left, spacing=spacing, direction=direction)
            if i > 0:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if do_preprocessing == True:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=decropped_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)
            else:
                probability_image = decropped_image

            if j == 0:  # not flipped
                probability_images_left.append(probability_image)
            else:
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image_flipped = ants.from_numpy(probability_array_flipped, 
                    origin=probability_image.origin, spacing=probability_image.spacing, 
                    direction=probability_image.direction)
                probability_images_right.append(probability_image_flipped)

    ################################
    #
    # Left:  do prediction of foreground and normalize to native space
    #
    ################################

    for j in range(2):
        probability_image = \
            ants.from_numpy(np.squeeze(predicted_data[1][j, :, :, :, 0]),
            origin=origin_left, spacing=spacing, direction=direction)
        decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)

        if do_preprocessing == True:
            probability_image = ants.apply_transforms(fixed=t1,
                moving=decropped_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", verbose=verbose)
        else:
            probability_image = decropped_image

        if j == 0:  # not flipped
            foreground_probability_image_left = probability_image
        else:
            probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
            probability_image_flipped = ants.from_numpy(probability_array_flipped, 
                origin=probability_image.origin, spacing=probability_image.spacing, 
                direction=probability_image.direction)
            foreground_probability_image_right = probability_image_flipped

    ################################
    #
    # Right:  build model and load weights
    #
    ################################

    channel_size = 1 + len(labels_right)
    number_of_classification_labels = 1 + len(labels_right)

    network_name = 'deepFlashRightT1'
    if t2 is not None:
        network_name = 'deepFlashRightBoth'
        channel_size += 1

    unet_model = create_unet_model_3d((*image_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0, weight_decay=0)

    penultimate_layer = unet_model.layers[-2].output
    output = Conv3D(filters=1,
                    kernel_size=(1, 1, 1),
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)
    unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output])

    if verbose == True:
        print("DeepFlash: retrieving model weights (right).")
    weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Right:  do prediction and normalize to native space
    #
    ################################

    if verbose == True:
        print("Prediction (right).")

    batchX = np.zeros((2, *image_size, channel_size))

    t1_cropped = ants.crop_indices(t1_preprocessed, lower_bound_right, upper_bound_right)
    batchX[0,:,:,:,0] = t1_cropped.numpy()
    t1_cropped = ants.crop_indices(t1_preprocessed_flipped, lower_bound_right, upper_bound_right)
    batchX[1,:,:,:,0] = t1_cropped.numpy()
    if t2 is not None:
        t2_cropped = ants.crop_indices(t2_preprocessed, lower_bound_right, upper_bound_right)
        batchX[0,:,:,:,1] = t2_cropped.numpy()
        t2_cropped = ants.crop_indices(t2_preprocessed_flipped, lower_bound_right, upper_bound_right)
        batchX[1,:,:,:,1] = t2_cropped.numpy()

    for i in range(len(priors_image_right_list)):
        cropped_prior = ants.crop_indices(priors_image_right_list[i], lower_bound_right, upper_bound_right)
        batchX[0,:,:,:,i + (channel_size - len(labels_right))] = cropped_prior.numpy()
        batchX[1,:,:,:,i + (channel_size - len(labels_right))] = cropped_prior.numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    for i in range(1 + len(labels_right)):
        for j in range(2):
            probability_image = \
                ants.from_numpy(np.squeeze(predicted_data[0][j, :, :, :, i]),
                origin=origin_right, spacing=spacing, direction=direction)
            if i > 0:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
            else:
                decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

            if do_preprocessing == True:
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=decropped_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose)
            else:
                probability_image = decropped_image

            if j == 0:  # not flipped
                probability_images_right[i] = (probability_images_right[i] + probability_image) / 2
            else:
                probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
                probability_image_flipped = ants.from_numpy(probability_array_flipped, 
                    origin=probability_image.origin, spacing=probability_image.spacing, 
                    direction=probability_image.direction)
                probability_images_left[i] = (probability_images_left[i] + probability_image_flipped) / 2

    ################################
    #
    # Right:  do prediction of foreground and normalize to native space
    #
    ################################

    for j in range(2):
        probability_image = \
            ants.from_numpy(np.squeeze(predicted_data[1][j, :, :, :, 0]),
            origin=origin_right, spacing=spacing, direction=direction)
        decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)

        if do_preprocessing == True:
            probability_image = ants.apply_transforms(fixed=t1,
                moving=decropped_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", verbose=verbose)
        else:
            probability_image = decropped_image

        if j == 0:  # not flipped
            foreground_probability_image_right = (foreground_probability_image_right + probability_image) / 2
        else:
            probability_array_flipped = np.flip(probability_image.numpy(), axis=0)
            probability_image_flipped = ants.from_numpy(probability_array_flipped, 
                origin=probability_image.origin, spacing=probability_image.spacing, 
                direction=probability_image.direction)
            foreground_probability_image_left = (foreground_probability_image_left + probability_image_flipped) / 2

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

    foreground_probability_image = foreground_probability_image_left + foreground_probability_image_right

    return_dict = {'segmentation_image' : relabeled_image,
                   'probability_images' : probability_images,
                   'foreground_probability_image' : foreground_probability_image}
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
    from ..utilities import categorical_focal_loss
    from ..utilities import preprocess_brain_image
    from ..utilities import crop_image_center
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
    if do_preprocessing == True:
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
            weight_decay=1e-5, additional_options=("attentionGating"))

        if verbose == True:
            print("DeepFlash: retrieving model weights.")

        weights_file_name = get_pretrained_network("deepFlash", antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose == True:
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

            if do_preprocessing == True:
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
            weight_decay = 1e-5, additional_options=("attentionGating"))

        if verbose == True:
            print("DeepFlash: retrieving model weights (left).")
        weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Left:  do prediction and normalize to native space
        #
        ################################

        if verbose == True:
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

            if do_preprocessing == True:
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
            weight_decay = 1e-5, additional_options=("attentionGating"))

        weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Right:  do prediction and normalize to native space
        #
        ################################

        if verbose == True:
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

            if do_preprocessing == True:
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


