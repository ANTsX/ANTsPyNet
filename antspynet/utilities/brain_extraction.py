import os
import shutil
import numpy as np
import keras

import requests
import tempfile
import sys

import ants


def brain_extraction(image,
                     modality="t1",
                     output_directory=None,
                     verbose=None):

    """
    Perform brain extraction using U-net and ANTs-based training data.  "NoBrainer" 
    is also possible where brain extraction uses U-net and FreeSurfer training data 
    ported from the

    https://github.com/neuronets/nobrainer-models

    Arguments
    ---------
    image : ANTsImage
        input image

    modality : string
        Modality image type.  Options include "t1", "bold", "fa", "t1nobrainer".

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
    >>> probability_brain_mask = brain_extraction(brain_image, modality="t1")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..architectures import create_nobrainer_unet_model_3d

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )  

    classes = ("background", "brain")
    number_of_classification_labels = len(classes)

    image_mods = [modality]
    channel_size = len(image_mods)

    if modality != "t1nobrainer":

        #####################
        #
        # ANTs-based
        #
        ##################### 

        weights_file_name = None
        if modality == "t1":
            if output_directory is not None:
                weights_file_name = output_directory + "/brainExtractionWeights.h5"
                if not os.path.exists(weights_file_name):
                    if verbose == True:
                        print("Brain extraction:  downloading weights.")
                    weights_file_name = get_pretrained_network("brainExtraction", weights_file_name)
            else:    
                weights_file_name = get_pretrained_network("brainExtraction")
        elif modality == "bold":
            if output_directory is not None:
                weights_file_name = output_directory + "/brainExtractionBoldWeights.h5"
                if not os.path.exists(weights_file_name):
                    if verbose == True:
                        print("Brain extraction:  downloading weights.")
                    weights_file_name = get_pretrained_network("brainExtractionBOLD", weights_file_name)
            else:    
                weights_file_name = get_pretrained_network("brainExtractionBOLD")
        elif modality == "fa":
            if output_directory is not None:
                weights_file_name = output_directory + "/brainExtractionFaWeights.h5"
                if not os.path.exists(weights_file_name):
                    if verbose == True:
                        print("Brain extraction:  downloading weights.")
                    weights_file_name = get_pretrained_network("brainExtractionFA", weights_file_name)
            else:    
                weights_file_name = get_pretrained_network("brainExtractionFA")
        else:
            raise ValueError("Unknown modality type.")    

        reorient_template_file_name = None
        reorient_template_file_exists = False
        if output_directory is not None:
            reorient_template_file_name = output_directory + "/S_template3_resampled.nii.gz"
            if os.path.exists(reorient_template_file_name):
                reorient_template_file_exists = True

        reorient_template = None
        if output_directory is None or reorient_template_file_exists == False:
            reorient_template_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")
            reorient_template_file.close()
            template_file_name = reorient_template_file.name
            template_url = "https://ndownloader.figshare.com/files/22597175"

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
    
    else:    

        #####################
        #
        # NoBrainer
        #
        ##################### 

        if verbose == True:
            print("NoBrainer:  generating network.")

        model = create_nobrainer_unet_model_3d((None, None, None, 1))

        weights_file_name = None
        if output_directory is not None:
            weights_file_name = output_directory + "/noBrainerWeights.h5"
            if not os.path.exists(weights_file_name):
                if verbose == True:
                    print("Brain extraction:  downloading weights.")
                weights_file_name = get_pretrained_network("brainExtractionNoBrainer", weights_file_name)
        else:    
            weights_file_name = get_pretrained_network("brainExtractionNoBrainer")
        model.load_weights(weights_file_name)

        if verbose == True:
            print("NoBrainer:  preprocessing (intensity truncation and resampling).")
          
        image_array = image.numpy()
        image_robust_range = np.quantile(image_array[np.where(image_array != 0)], (0.02, 0.98))
        threshold_value = 0.10 * (image_robust_range[1] - image_robust_range[0]) + image_robust_range[0]

        thresholded_mask = ants.threshold_image(image, -10000, threshold_value, 0, 1)
        thresholded_image = image * thresholded_mask

        image_resampled = ants.resample_image(thresholded_image, (256, 256, 256), use_voxels=True)
        image_array = np.expand_dims(image_resampled.numpy(), axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        if verbose == True:
            print("NoBrainer:  predicting mask.")

        brain_mask_array = np.squeeze(model.predict(image_array, verbose=verbose))
        brain_mask_resampled = ants.copy_image_info(image_resampled, ants.from_numpy(brain_mask_array))
        brain_mask_image = ants.resample_image(brain_mask_resampled, image.shape, use_voxels=True, interp_type=1)
       
        spacing = ants.get_spacing(image)
        spacing_product = spacing[0] * spacing[1] * spacing[2]
        minimum_brain_volume = round(649933.7/spacing_product)
        brain_mask_labeled = ants.label_clusters(brain_mask_image, minimum_brain_volume)

        return(brain_mask_labeled)

