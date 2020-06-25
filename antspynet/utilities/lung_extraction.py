import os
import shutil
import numpy as np
import keras

import requests
import tempfile
import sys

import ants


def lung_extraction(image,
                    modality="proton",
                    output_directory=None,
                    verbose=None):

    """
    Perform proton or ct lung extraction using U-net.

    Arguments
    ---------
    image : ANTsImage
        input image

    modality : string
        Modality image type.  Options include "ct" and "proton".

    output_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        tempfile.

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

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )


    image_mods = [modality]
    channel_size = len(image_mods)

    weights_file_name = None
    unet_model = None

    if modality == "proton":
        if output_directory is not None:
            weights_file_name = output_directory + "/protonLungSegmentationWeights.h5"
            if not os.path.exists(weights_file_name):
                if verbose == True:
                    print("Lung extraction:  downloading weights.")
                weights_file_name = get_pretrained_network("protonLungMri", weights_file_name)
        else:
            weights_file_name = get_pretrained_network("protonLungMri")

        classes = ("background", "left_lung", "right_lung")
        number_of_classification_labels = len(classes)

        reorient_template_file_name = None
        reorient_template_file_exists = False
        if output_directory is not None:
            reorient_template_file_name = output_directory + "/protonLungTemplate.nii.gz"
            if os.path.exists(reorient_template_file_name):
                reorient_template_file_exists = True

        reorient_template = None
        if output_directory is None or reorient_template_file_exists == False:
            reorient_template_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            reorient_template_file.close()
            template_file_name = reorient_template_file.name
            template_url = "https://ndownloader.figshare.com/files/22707338"

            if not os.path.exists(template_file_name):
                if verbose == True:
                    print("Lung extraction:  downloading template.")
                r = requests.get(template_url)
                with open(template_file_name, 'wb') as f:
                    f.write(r.content)
            reorient_template = ants.image_read(template_file_name)
            if output_directory is not None:
                shutil.copy(template_file_name, reorient_template_file_name)
        else:
            reorient_template = ants.image_read(reorient_template_file_name)

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
        if output_directory is not None:
            weights_file_name = output_directory + "/ctLungSegmentationWeights.h5"
            if not os.path.exists(weights_file_name):
                if verbose == True:
                    print("Lung extraction:  downloading weights.")
                weights_file_name = get_pretrained_network("ctHumanLung", weights_file_name)
        else:
            weights_file_name = get_pretrained_network("ctHumanLung")

        classes = ("background", "left_lung", "right_lung", "trachea")
        number_of_classification_labels = len(classes)

        reorient_template_file_name = None
        reorient_template_file_exists = False
        if output_directory is not None:
            reorient_template_file_name = output_directory + "/ctLungTemplate.nii.gz"
            if os.path.exists(reorient_template_file_name):
                reorient_template_file_exists = True

        reorient_template = None
        if output_directory is None or reorient_template_file_exists == False:
            reorient_template_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            reorient_template_file.close()
            template_file_name = reorient_template_file.name
            template_url = "https://ndownloader.figshare.com/files/22707335"

            if not os.path.exists(template_file_name):
                if verbose == True:
                    print("Lung extraction:  downloading template.")
                r = requests.get(template_url)
                with open(template_file_name, 'wb') as f:
                    f.write(r.content)
            reorient_template = ants.image_read(template_file_name)
            if output_directory is not None:
                shutil.copy(template_file_name, reorient_template_file_name)
        else:
            reorient_template = ants.image_read(reorient_template_file_name)

        resampled_image_size = reorient_template.shape

        unet_model = create_unet_model_3d((*resampled_image_size, channel_size),
            number_of_outputs = number_of_classification_labels,
            number_of_layers = 4, number_of_filters_at_base_layer = 8, dropout_rate = 0.0,
            convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2))
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





