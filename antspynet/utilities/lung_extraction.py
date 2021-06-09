import numpy as np
import tensorflow as tf
import ants

def lung_extraction(image,
                    modality="proton",
                    antsxnet_cache_directory=None,
                    verbose=None):

    """
    Perform proton or ct lung extraction using U-net.

    Arguments
    ---------
    image : ANTsImage
        input image

    modality : string
        Modality image type.  Options include "ct" and "proton".

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dictionary of ANTs segmentation and probability images.

    Example
    -------
    >>> output = lung_extraction(lung_image, modality="proton")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    image_mods = [modality]
    channel_size = len(image_mods)

    weights_file_name = None
    unet_model = None

    if modality == "proton":
        weights_file_name = get_pretrained_network("protonLungMri",
            antsxnet_cache_directory=antsxnet_cache_directory)

        classes = ("background", "left_lung", "right_lung")
        number_of_classification_labels = len(classes)

        reorient_template_file_name_path = get_antsxnet_data("protonLungTemplate",
            antsxnet_cache_directory=antsxnet_cache_directory)
        reorient_template = ants.image_read(reorient_template_file_name_path)

        resampled_image_size = reorient_template.shape

        unet_model = create_unet_model_3d((*resampled_image_size, channel_size),
            number_of_outputs = number_of_classification_labels,
            number_of_layers = 4, number_of_filters_at_base_layer = 16, dropout_rate = 0.0,
            convolution_kernel_size = (7, 7, 5), deconvolution_kernel_size = (7, 7, 5))
        unet_model.load_weights(weights_file_name)

        if verbose == True:
            print("Lung extraction:  normalizing image to the template.")

        center_of_mass_template = ants.get_center_of_mass(reorient_template * 0 + 1)
        center_of_mass_image = ants.get_center_of_mass(image * 0 + 1)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)
        warped_image = ants.apply_ants_transform_to_image(xfrm, image, reorient_template)

        batchX = np.expand_dims(warped_image.numpy(), axis=0)
        batchX = np.expand_dims(batchX, axis=-1)
        batchX = (batchX - batchX.mean()) / batchX.std()

        predicted_data = unet_model.predict(batchX, verbose=0)

        origin = warped_image.origin
        spacing = warped_image.spacing
        direction = warped_image.direction

        probability_images_array = list()
        for i in range(number_of_classification_labels):
            probability_images_array.append(
            ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
                origin=origin, spacing=spacing, direction=direction))

        if verbose == True:
            print("Lung extraction:  renormalize probability mask to native space.")

        for i in range(number_of_classification_labels):
            probability_images_array[i] = ants.apply_ants_transform_to_image(
                ants.invert_ants_transform(xfrm), probability_images_array[i], image)

        image_matrix = ants.image_list_to_matrix(probability_images_array, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        return_dict = {'segmentation_image' : segmentation_image,
                       'probability_images' : probability_images_array}
        return(return_dict)

    elif modality == "ct":

        ################################
        #
        # Preprocess image
        #
        ################################

        if verbose == True:
            print("Preprocess CT image.")

        def closest_simplified_direction_matrix(direction):
            closest = np.floor(np.abs(direction + 0.5))
            closest[direction < 0] *= -1.0
            return direction

        simplified_direction = closest_simplified_direction_matrix(image.direction)

        reference_image_size = (128, 128, 128)

        ct_preprocessed = ants.resample_image(image, reference_image_size, use_voxels=True, interp_type=0)
        ct_preprocessed[ct_preprocessed < -1000] = -1000
        ct_preprocessed[ct_preprocessed > 400] = 400
        ct_preprocessed.set_direction(simplified_direction)
        ct_preprocessed.set_origin((0, 0, 0))
        ct_preprocessed.set_spacing((1, 1, 1))

        ################################
        #
        # Reorient image
        #
        ################################

        reference_image = ants.make_image(reference_image_size,
                                          voxval=0,
                                          spacing=(1, 1, 1),
                                          origin=(0, 0, 0),
                                          direction=np.identity(3))
        center_of_mass_reference = np.floor(ants.get_center_of_mass(reference_image * 0 + 1))
        center_of_mass_image = np.floor(ants.get_center_of_mass(ct_preprocessed * 0 + 1))
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_reference), translation=translation)
        ct_preprocessed = ((ct_preprocessed - ct_preprocessed.min()) /
            (ct_preprocessed.max() - ct_preprocessed.min()))
        ct_preprocessed_warped = ants.apply_ants_transform_to_image(
            xfrm, ct_preprocessed, reference_image, interpolation="nearestneighbor")
        ct_preprocessed_warped = ((ct_preprocessed_warped - ct_preprocessed_warped.min()) /
            (ct_preprocessed_warped.max() - ct_preprocessed_warped.min())) - 0.5

        ################################
        #
        # Build models and load weights
        #
        ################################

        if verbose == True:
            print("Build model and load weights.")

        weights_file_name = get_pretrained_network("lungCtWithPriorsSegmentationWeights", 
            antsxnet_cache_directory=antsxnet_cache_directory)

        classes = ("background", "left lung", "right lung", "airways")
        number_of_classification_labels = len(classes)

        luna16_priors = ants.ndimage_to_list(ants.image_read(get_antsxnet_data("luna16LungPriors")))
        for i in range(len(luna16_priors)):
            luna16_priors[i] = ants.resample_image(luna16_priors[i], reference_image_size, use_voxels=True)
        channel_size = len(luna16_priors) + 1

        unet_model = create_unet_model_3d((*reference_image_size, channel_size),
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            weight_decay=1e-5, additional_options=("attentionGating",))
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose == True:
            print("Prediction.")

        batchX = np.zeros((1, *reference_image_size, channel_size))
        batchX[:,:,:,:,0] = ct_preprocessed_warped.numpy()

        for i in range(len(luna16_priors)):
            batchX[:,:,:,:,i+1] = luna16_priors[i].numpy() - 0.5

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        probability_images = list()
        for i in range(number_of_classification_labels):
            if verbose == True:
                print("Reconstructing image", classes[i])
            probability_image = ants.from_numpy(np.squeeze(predicted_data[:,:,:,:,i]), 
                origin=ct_preprocessed_warped.origin, spacing=ct_preprocessed_warped.spacing,
                direction=ct_preprocessed_warped.direction)    
            probability_image = ants.apply_ants_transform_to_image(
                ants.invert_ants_transform(xfrm), probability_image, ct_preprocessed)
            probability_image = ants.resample_image(probability_image,
               resample_params=image.shape, use_voxels=True, interp_type=0)
            probability_image = ants.copy_image_info(image, probability_image)
            probability_images.append(probability_image)

        image_matrix = ants.image_list_to_matrix(probability_images, image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), image * 0 + 1)[0]

        return_dict = {'segmentation_image' : segmentation_image,
                       'probability_images' : probability_images}
        return(return_dict)





