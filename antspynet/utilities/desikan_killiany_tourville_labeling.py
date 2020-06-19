import os

import numpy as np
import keras

import ants

def desikan_killiany_tourville_labeling(t1,
                                        do_preprocessing=True,
                                        output_directory=None,
                                        verbose=False):

    """
    Cortical and deep gray matter labeling using Desikan-Killiany-Tourville

    Perform Atropos-style six tissue segmentation using deep learning

    The labeling is as follows:
    Label 0: background
    Label 4: left lateral ventricle
    Label 5: left inferior lateral ventricle
    Label 6: left cerebellem exterior
    Label 7: left cerebellum white matter
    Label 10: left thalamus proper
    Label 11: left caudate
    Label 12: left putamen
    Label 13: left pallidium
    Label 15: 4th ventricle
    Label 16: brain stem
    Label 17: left hippocampus
    Label 18: left amygdala
    Label 24: CSF
    Label 25: left lesion
    Label 26: left accumbens area
    Label 28: left ventral DC
    Label 30: left vessel
    Label 43: right lateral ventricle
    Label 44: right inferior lateral ventricle
    Label 45: right cerebellum exterior
    Label 46: right cerebellum white matter
    Label 49: right thalamus proper
    Label 50: right caudate
    Label 51: right putamen
    Label 52: right palladium
    Label 53: right hippocampus
    Label 54: right amygdala
    Label 57: right lesion
    Label 58: right accumbens area
    Label 60: right ventral DC
    Label 62: right vessel
    Label 72: 5th ventricle
    Label 85: optic chasm
    Label 91: left basal forebrain
    Label 92: right basal forebrain
    Label 630: cerebellar vermal lobules I-V
    Label 631: cerebellar vermal lobules VI-VII
    Label 632: cerebellar vermal lobules VIII-X
    Label 1002: left caudal anterior cingulate
    Label 1003: left caudal middle frontal
    Label 1005: left cuneus
    Label 1006: left entorhinal
    Label 1007: left fusiform
    Label 1008: left inferior parietal
    Label 1009: left inferior temporal
    Label 1010: left isthmus cingulate
    Label 1011: left lateral occipital
    Label 1012: left lateral orbitofrontal
    Label 1013: left lingual
    Label 1014: left medial orbitofrontal
    Label 1015: left middle temporal
    Label 1016: left parahippocampal
    Label 1017: left paracentral
    Label 1018: left pars opercularis
    Label 1019: left pars orbitalis
    Label 1020: left pars triangularis
    Label 1021: left pericalcarine
    Label 1022: left postcentral
    Label 1023: left posterior cingulate
    Label 1024: left precentral
    Label 1025: left precuneus
    Label 1026: left rostral anterior cingulate
    Label 1027: left rostral middle frontal
    Label 1028: left superior frontal
    Label 1029: left superior parietal
    Label 1030: left superior temporal
    Label 1031: left supramarginal
    Label 1034: left transverse temporal
    Label 1035: left insula
    Label 2002: right caudal anterior cingulate
    Label 2003: right caudal middle frontal
    Label 2005: right cuneus
    Label 2006: right entorhinal
    Label 2007: right fusiform
    Label 2008: right inferior parietal
    Label 2009: right inferior temporal
    Label 2010: right isthmus cingulate
    Label 2011: right lateral occipital
    Label 2012: right lateral orbitofrontal
    Label 2013: right lingual
    Label 2014: right medial orbitofrontal
    Label 2015: right middle temporal
    Label 2016: right parahippocampal
    Label 2017: right paracentral
    Label 2018: right pars opercularis
    Label 2019: right pars orbitalis
    Label 2020: right pars triangularis
    Label 2021: right pericalcarine
    Label 2022: right postcentral
    Label 2023: right posterior cingulate
    Label 2024: right precentral
    Label 2025: right precuneus
    Label 2026: right rostral anterior cingulate
    Label 2027: right rostral middle frontal
    Label 2028: right superior frontal
    Label 2029: right superior parietal
    Label 2030: right superior temporal
    Label 2031: right supramarginal
    Label 2034: right transverse temporal
    Label 2035: right insula

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
    >>> flash = desikan_killiany_tourville_labeling(image)
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
        weights_file_name = output_directory + "/dktLabelingInner.h5"
        if not os.path.exists(weights_file_name):
            if verbose == True:
                print("DesikianKillianyTourville:  downloading model weights.")
            weights_file_name = get_pretrained_network("dktInner", weights_file_name)
    else:
        weights_file_name = get_pretrained_network("dktInner")

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
