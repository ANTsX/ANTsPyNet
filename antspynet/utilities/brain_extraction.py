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
        input image (or list of images for multi-modal scenarios).

    modality : string
        Modality image type.  Options include:
            * "t1": T1-weighted MRI---ANTs-trained.
            * "t1nobrainer": T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.
            * "t1combined": Brian's combination of "t1" and "t1nobrainer".  One can also specify
                            "t1combined[X]" where X is the morphological radius.  X = 12 by default.
            * "flair": FLAIR MRI.
            * "t2": T2 MRI.
            * "bold": 3-D BOLD MRI.
            * "fa": Fractional anisotropy.
            * "t1t2infant": Combined T1-w/T2-w infant MRI h/t Martin Styner.
            * "t1infant": T1-w infant MRI h/t Martin Styner.
            * "t2infant": T2-w infant MRI h/t Martin Styner.

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

    classes = ("background", "brain")
    number_of_classification_labels = len(classes)

    channel_size = 1
    if isinstance(image, list):
        channel_size = len(image)

    input_images = list()
    if channel_size == 1:
        input_images.append(image)
    else:
        input_images = image

    if input_images[0].dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if "t1combined" in modality:

        brain_extraction_t1 = brain_extraction(image, modality="t1",
          output_directory=output_directory, verbose=verbose)
        brain_mask = ants.iMath_get_largest_component(
          ants.threshold_image(brain_extraction_t1, 0.5, 10000))

        # Need to change with voxel resolution
        morphological_radius = 12
        if '[' in modality and ']' in modality:
            morphological_radius = int(modality.split("[")[1].split("]")[0])

        brain_extraction_t1nobrainer = brain_extraction(image * ants.iMath_MD(brain_mask, radius=morphological_radius),
          modality = "t1nobrainer", output_directory=output_directory, verbose=verbose)
        brain_extraction_combined = ants.iMath_fill_holes(
          ants.iMath_get_largest_component(brain_extraction_t1nobrainer * brain_mask))

        brain_extraction_combined = brain_extraction_combined + ants.iMath_ME(brain_mask, morphological_radius) + brain_mask

        return(brain_extraction_combined)

    if modality != "t1nobrainer":

        #####################
        #
        # ANTs-based
        #
        #####################

        weights_file_name = None
        weights_file_name_prefix = None

        if modality == "t1":
            weights_file_name = "/brainExtractionWeights.h5"
            weights_file_name_prefix = "brainExtraction"
        elif modality == "bold":
            weights_file_name = "/brainExtractionBoldWeights.h5"
            weights_file_name_prefix = "brainExtractionBOLD"
        elif modality == "t2":
            weights_file_name = "/brainExtractionT2Weights.h5"
            weights_file_name_prefix = "brainExtractionT2"
        elif modality == "flair":
            weights_file_name = "/brainExtractionFlairWeights.h5"
            weights_file_name_prefix = "brainExtractionFLAIR"
        elif modality == "fa":
            weights_file_name = "/brainExtractionFaWeights.h5"
            weights_file_name_prefix = "brainExtractionFA"
        elif modality == "t1t2infant":
            weights_file_name = "/brainExtractionInfantT1T2Weights.h5"
            weights_file_name_prefix = "brainExtractionInfantT1T2"
        elif modality == "t1infant":
            weights_file_name = "/brainExtractionInfantT1Weights.h5"
            weights_file_name_prefix = "brainExtractionInfantT1"
        elif modality == "t2infant":
            weights_file_name = "/brainExtractionInfantT2Weights.h5"
            weights_file_name_prefix = "brainExtractionInfantT2"
        else:
            raise ValueError("Unknown modality type.")

        if output_directory is not None:
            weights_file_name = output_directory + weights_file_name

        if output_directory is not None:
            if not os.path.exists(weights_file_name):
                if verbose == True:
                    print("Brain extraction:  downloading weights.")
                weights_file_name = get_pretrained_network(weights_file_name_prefix, weights_file_name)
        else:
            weights_file_name = get_pretrained_network(weights_file_name_prefix)

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
        center_of_mass_image = ants.get_center_of_mass(input_images[0])
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)

        batchX = np.zeros((1, *resampled_image_size, channel_size))
        for i in range(len(input_images)):
            warped_image = ants.apply_ants_transform_to_image(xfrm, input_images[i], reorient_template)
            warped_array = warped_image.numpy()
            batchX[0,:,:,:,i] = (warped_array - warped_array.mean()) / warped_array.std()

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
            ants.invert_ants_transform(xfrm), probability_images_array[1], input_images[0])

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

