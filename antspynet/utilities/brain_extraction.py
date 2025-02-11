import numpy as np
import tensorflow as tf

import ants

def brain_extraction(image,
                     modality,
                     verbose=False):

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
            * "t1": T1-weighted MRI---ANTs-trained.  Previous versions are specified as "t1.v0", "t1.v1".
            * "t1nobrainer": T1-weighted MRI---FreeSurfer-trained: h/t Satra Ghosh and Jakub Kaczmarzyk.
            * "t1combined": Brian's combination of "t1" and "t1nobrainer".  One can also specify
                            "t1combined[X]" where X is the morphological radius.  X = 12 by default.
            * "t1threetissue":  T1-weighted MRI---originally developed from BrainWeb20 (and later
                                expanded).  Label 1: brain + subdural CSF, label 2: sinuses + skull,
                                label 3: other head, face, neck tissue.
            * "t1hemi":  Label 1 of "t1threetissue" subdivided into left and right hemispheres.
            * "t1lobes":  Labels 1) frontal, 2) parietal, 3) temporal, 4) occipital. 5) csf,
              cerebellum, and brain stem.
            * "flair": FLAIR MRI.   Previous versions are specified as "flair.v0".
            * "t2": T2 MRI.  Previous versions are specified as "t2.v0".
            * "t2star": T2Star MRI.
            * "bold": 3-D mean BOLD MRI.  Previous versions are specified as "bold.v0".
            * "fa": fractional anisotropy.  Previous versions are specified as "fa.v0".
            * "mra": MRA h/t Tyler Hanson "mmbop".
            * "t1t2infant": Combined T1-w/T2-w infant MRI h/t Martin Styner.
            * "t1infant": T1-w infant MRI h/t Martin Styner.
            * "t2infant": T2-w infant MRI h/t Martin Styner.

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
    from ..utilities import get_antsxnet_data
    from ..architectures import create_nobrainer_unet_model_3d
    from ..utilities import decode_unet
    from ..utilities import pad_or_crop_image_to_size

    channel_size = 1
    if isinstance(image, list):
        channel_size = len(image)

    input_images = list()
    if channel_size == 1:
        if modality == "t1hemi" or modality == "t1lobes":
            bext = brain_extraction(image, modality="t1threetissue", verbose=verbose)
            mask = ants.threshold_image(bext['segmentation_image'], 1, 1, 1, 0)
            input_images.append(image * mask)
        else:
            input_images.append(image)
    else:
        input_images = image

    if input_images[0].dimension != 3:
        raise ValueError("Image dimension must be 3.")

    for i in range(len(input_images)):
        if input_images[i].pixeltype != 'float':
            input_images[i] = input_images[i].clone('float')

    if "t1combined" in modality:
        # Need to change with voxel resolution
        morphological_radius = 12
        if '[' in modality and ']' in modality:
            morphological_radius = int(modality.split("[")[1].split("]")[0])

        brain_extraction_t1 = brain_extraction(image, modality="t1", verbose=verbose)
        brain_mask = ants.iMath_get_largest_component(
          ants.threshold_image(brain_extraction_t1, 0.5, 10000))
        brain_mask = ants.morphology(brain_mask, "close", morphological_radius).iMath_fill_holes()

        brain_extraction_t1nobrainer = brain_extraction(image * ants.iMath_MD(brain_mask, radius=morphological_radius),
          modality = "t1nobrainer", verbose=verbose)
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

        weights_file_name_prefix = None
        is_standard_network = False

        if modality == "t1.v0":
            weights_file_name_prefix = "brainExtraction"
        elif modality == "t1.v1":
            weights_file_name_prefix = "brainExtractionT1v1"
            is_standard_network = True
        elif modality == "t1":
            weights_file_name_prefix = "brainExtractionRobustT1"
            is_standard_network = True
        elif modality == "t2.v0":
            weights_file_name_prefix = "brainExtractionT2"
        elif modality == "t2":
            weights_file_name_prefix = "brainExtractionRobustT2"
            is_standard_network = True
        elif modality == "t2star":
            weights_file_name_prefix = "brainExtractionRobustT2Star"
            is_standard_network = True
        elif modality == "flair.v0":
            weights_file_name_prefix = "brainExtractionFLAIR"
        elif modality == "flair":
            weights_file_name_prefix = "brainExtractionRobustFLAIR"
            is_standard_network = True
        elif modality == "bold.v0":
            weights_file_name_prefix = "brainExtractionBOLD"
        elif modality == "bold":
            weights_file_name_prefix = "brainExtractionRobustBOLD"
            is_standard_network = True
        elif modality == "fa.v0":
            weights_file_name_prefix = "brainExtractionFA"
        elif modality == "fa":
            weights_file_name_prefix = "brainExtractionRobustFA"
            is_standard_network = True
        elif modality == "mra":
            weights_file_name_prefix = "brainExtractionMra"
            is_standard_network = True
        elif modality == "t1t2infant":
            weights_file_name_prefix = "brainExtractionInfantT1T2"
        elif modality == "t1infant":
            weights_file_name_prefix = "brainExtractionInfantT1"
        elif modality == "t2infant":
            weights_file_name_prefix = "brainExtractionInfantT2"
        elif modality == "t1threetissue":
            weights_file_name_prefix = "brainExtractionBrainWeb20"
            is_standard_network = True
        elif modality == "t1hemi":
            weights_file_name_prefix = "brainExtractionT1Hemi"
            is_standard_network = True
        elif modality == "t1lobes":
            weights_file_name_prefix = "brainExtractionT1Lobes"
            is_standard_network = True
        else:
            raise ValueError("Unknown modality type.")

        if verbose:
            print("Brain extraction:  retrieving model weights.")

        weights_file_name = get_pretrained_network(weights_file_name_prefix)

        if verbose:
            print("Brain extraction:  retrieving template.")

        if modality == "t1threetissue":
            reorient_template = ants.image_read(get_antsxnet_data("nki"))
        elif modality == "t1hemi" or modality == "t1lobes":
            reorient_template = ants.image_read(get_antsxnet_data("hcpyaT1Template"))
            reorient_template_mask = ants.image_read(get_antsxnet_data("hcpyaTemplateBrainMask"))
            reorient_template = reorient_template * reorient_template_mask
            reorient_template = ants.resample_image(reorient_template, (1, 1, 1), use_voxels=False, interp_type=0)
            reorient_template = pad_or_crop_image_to_size(reorient_template, (160, 192, 160))
            xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
                center=np.asarray(ants.get_center_of_mass(reorient_template)), translation=(0, 0, -10))
            reorient_template = xfrm.apply_to_image(reorient_template)
        else:
            reorient_template = ants.image_read(get_antsxnet_data("S_template3"))
            if is_standard_network and (modality != "t1.v1" and modality != "mra"):
                ants.set_spacing(reorient_template, (1.5, 1.5, 1.5))
        resampled_image_size = reorient_template.shape

        number_of_filters = (8, 16, 32, 64)
        number_of_classification_labels = 2
        mode = "classification"
        if is_standard_network:
            number_of_filters = (16, 32, 64, 128)
            number_of_classification_labels = 1
            mode = "sigmoid"

        unet_model = None
        if modality == "t1threetissue" or modality == "t1hemi" or modality == "t1lobes":
            mode = "classification"
            if modality == "t1threetissue":
                number_of_classification_labels = 4 # background, brain, meninges/csf, misc. head
            elif modality == "t1hemi":
                number_of_classification_labels = 3 # background, left, right
            elif modality == "t1lobes":
                number_of_classification_labels = 6 # background, frontal, parietal, temporal, occipital, misc
            unet_model = create_unet_model_3d((*resampled_image_size, channel_size),
                number_of_outputs=number_of_classification_labels, mode=mode,
                number_of_filters=number_of_filters, dropout_rate=0.0,
                convolution_kernel_size=3, deconvolution_kernel_size=2,
                weight_decay=0)
        else:
            unet_model = create_unet_model_3d((*resampled_image_size, channel_size),
                number_of_outputs=number_of_classification_labels, mode=mode,
                number_of_filters=number_of_filters, dropout_rate=0.0,
                convolution_kernel_size=3, deconvolution_kernel_size=2,
                weight_decay=1e-5)

        unet_model.load_weights(weights_file_name)

        if verbose:
            print("Brain extraction:  normalizing image to the template.")

        center_of_mass_template = ants.get_center_of_mass(reorient_template)
        center_of_mass_image = ants.get_center_of_mass(input_images[0])
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)

        batchX = np.zeros((1, *resampled_image_size, channel_size))

        for i in range(len(input_images)):
            warped_image = ants.apply_ants_transform_to_image(xfrm, input_images[i], reorient_template)
            if is_standard_network and modality != "t1.v1":
                batchX[0,:,:,:,i] = (ants.iMath(warped_image, "Normalize")).numpy()
            else:
                warped_array = warped_image.numpy()
                batchX[0,:,:,:,i] = (warped_array - warped_array.mean()) / warped_array.std()

        if verbose:
            print("Brain extraction:  prediction and decoding.")

        predicted_data = unet_model.predict(batchX, verbose=verbose)
        probability_images = decode_unet(predicted_data, reorient_template)

        if verbose:
            print("Brain extraction:  renormalize probability mask to native space.")

        xfrm_inv = xfrm.invert()

        if modality == "t1threetissue" or modality == "t1hemi" or modality == "t1lobes":
            probability_images_warped = list()
            for i in range(number_of_classification_labels):
                probability_images_warped.append(xfrm_inv.apply_to_image(
                    probability_images[0][i], input_images[0]))

            image_matrix = ants.image_list_to_matrix(probability_images_warped, input_images[0] * 0 + 1)
            segmentation_matrix = np.argmax(image_matrix, axis=0)
            segmentation_image = ants.matrix_to_images(
                np.expand_dims(segmentation_matrix, axis=0), input_images[0] * 0 + 1)[0]

            return_dict = {'segmentation_image' : segmentation_image,
                           'probability_images' : probability_images_warped}
            return(return_dict)
        else:
            probability_image = xfrm_inv.apply_to_image(probability_images[0][number_of_classification_labels-1], input_images[0])
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
