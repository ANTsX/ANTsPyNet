import os
import shutil
import numpy as np
import keras

import requests
import tempfile
import sys

import ants


def brain_extraction(image,
                     output_directory=None,
                     verbose=None):

    """
    Perform brain extraction using U-net and ANTs-based
    training data.

    Arguments
    ---------
    image : ANTsImage
        input image

    output_directory : string
        Destination directory for storing the downloaded template and model weights.  
        Since these can be resused, if is None, these data will be downloaded to a 
        tempfile.

    verbose : boolean
        Print progress to the screen.    

    Returns
    -------
    ANTs probability brain mask image.

    Example
    -------
    >>> probability_brain_mask = brain_extraction(brain_image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network

    classes = ("background", "brain")
    number_of_classification_labels = len(classes)

    image_mods = ["T1"]
    channel_size = len(image_mods)

    reorient_template_file_name = None
    reorient_template_file_exists = False
    if output_directory is not None:
        reorient_template_file_name = output_directory + "S_template3_resampled.nii.gz"
        if os.path.exists(reorient_template_file_name):
            reorient_template_file_exists = True

    reorient_template = None
    if output_directory is None or reorient_template_file_exists == False:
        reorient_template_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")
        reorient_template_file.close()
        template_file_name = reorient_template_file.name
        template_url = "https://github.com/ANTsXNet/BrainExtraction/blob/master/Data/Template/S_template3_resampled.nii.gz?raw=true"

        if not os.path.exists(template_file_name):
            if verbose == True:
                print("Brain extraction:  downloading template.")
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
        convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
        weight_decay = 1e-5)

    weights_file_name = None
    if output_directory is not None:
        weights_file_name = output_directory + "brainExtractionWeights.h5"
        if verbose == True and not os.path.exists(weights_file_name):
            print("Brain extraction:  downloading template.")
            weights_file_name = get_pretrained_network("brainExtraction", weights_file_name)
    else:    
        if verbose == True:
            print("Brain extraction:  downloading template.")
        weights_file_name = get_pretrained_network("brainExtraction", weights_file_name)

    unet_model.load_weights(weights_file_name)

    if verbose == True:
        print("Brain extraction:  normalizing image to the template.")
    
    center_of_mass_template = ants.get_center_of_mass(reorient_template)
    center_of_mass_image = ants.get_center_of_mass(image)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)
    warped_image = ants.apply_ants_transform_to_image(xfrm, image, reorient_template)
    warped_image = (warped_image - warped_image.mean()) / warped_image.std()

    batchX = np.expand_dims(warped_image.numpy(), axis=0)
    batchX = np.expand_dims(batchX, axis=-1)
    batchX = (batchX - batchX.mean()) / batchX.std()

    if verbose == True:
        print("Brain extraction:  prediction and decoding.")

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    origin = reorient_template.origin
    spacing = reorient_template.spacing
    direction = reorient_template.direction

    probability_images_array = list()
    probability_images_array.append(
    ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, 0]),
        origin=origin, spacing=spacing, direction=direction))
    probability_images_array.append(
    ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, 1]),
        origin=origin, spacing=spacing, direction=direction))

    if verbose == True:
        print("Brain extraction:  renormalize probability mask to native space.")
    probability_image = ants.apply_ants_transform_to_image(
        ants.invert_ants_transform(xfrm), probability_images_array[1], image)

    return(probability_image)    
