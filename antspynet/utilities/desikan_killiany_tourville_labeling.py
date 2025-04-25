import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import ants

def desikan_killiany_tourville_labeling(t1,
                                        do_preprocessing=True,
                                        return_probability_images=False,
                                        do_lobar_parcellation=False,
                                        version=0,
                                        verbose=False):

    """
    Desikan-Killiany-Tourville labeling.  https://pubmed.ncbi.nlm.nih.gov/23227001/

    Perform DKT labeling using deep learning.  Description available here:
    https://mindboggle.readthedocs.io/en/latest/labels.html

    The labeling is as follows:

    Outer labels:
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

    Performing the lobar parcellation is based on the FreeSurfer division
    described here:

    See https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation

    Frontal lobe:
    Label 1002:  left caudal anterior cingulate
    Label 1003:  left caudal middle frontal
    Label 1012:  left lateral orbitofrontal
    Label 1014:  left medial orbitofrontal
    Label 1017:  left paracentral
    Label 1018:  left pars opercularis
    Label 1019:  left pars orbitalis
    Label 1020:  left pars triangularis
    Label 1024:  left precentral
    Label 1026:  left rostral anterior cingulate
    Label 1027:  left rostral middle frontal
    Label 1028:  left superior frontal
    Label 2002:  right caudal anterior cingulate
    Label 2003:  right caudal middle frontal
    Label 2012:  right lateral orbitofrontal
    Label 2014:  right medial orbitofrontal
    Label 2017:  right paracentral
    Label 2018:  right pars opercularis
    Label 2019:  right pars orbitalis
    Label 2020:  right pars triangularis
    Label 2024:  right precentral
    Label 2026:  right rostral anterior cingulate
    Label 2027:  right rostral middle frontal
    Label 2028:  right superior frontal

    Parietal:
    Label 1008:  left inferior parietal
    Label 1010:  left isthmus cingulate
    Label 1022:  left postcentral
    Label 1023:  left posterior cingulate
    Label 1025:  left precuneus
    Label 1029:  left superior parietal
    Label 1031:  left supramarginal
    Label 2008:  right inferior parietal
    Label 2010:  right isthmus cingulate
    Label 2022:  right postcentral
    Label 2023:  right posterior cingulate
    Label 2025:  right precuneus
    Label 2029:  right superior parietal
    Label 2031:  right supramarginal

    Temporal:
    Label 1006:  left entorhinal
    Label 1007:  left fusiform
    Label 1009:  left inferior temporal
    Label 1015:  left middle temporal
    Label 1016:  left parahippocampal
    Label 1030:  left superior temporal
    Label 1034:  left transverse temporal
    Label 2006:  right entorhinal
    Label 2007:  right fusiform
    Label 2009:  right inferior temporal
    Label 2015:  right middle temporal
    Label 2016:  right parahippocampal
    Label 2030:  right superior temporal
    Label 2034:  right transverse temporal

    Occipital:
    Label 1005:  left cuneus
    Label 1011:  left lateral occipital
    Label 1013:  left lingual
    Label 1021:  left pericalcarine
    Label 2005:  right cuneus
    Label 2011:  right lateral occipital
    Label 2013:  right lingual
    Label 2021:  right pericalcarine

    Other outer labels:
    Label 1035:  left insula
    Label 2035:  right insula

    For the original DKT version (version 0), the set of inner labels are
    also included.

    Inner labels:
    Label 0: background
    Label 4: left lateral ventricle
    Label 5: left inferior lateral ventricle
    Label 6: left cerebellem exterior
    Label 7: left cerebellum white matter
    Label 10: left thalamus proper
    Label 11: left caudate
    Label 12: left putamen
    Label 13: left pallidium
    Label 14: 3rd ventricle
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

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * brain extraction, and
       * affine registration to HCP.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    do_preprocessing : boolean
        See description above.

    return_probability_images : boolean
        Whether to return the two sets of probability images for the inner and outer
        labels.

    do_lobar_parcellation : boolean
        Perform lobar parcellation (also divided by hemisphere).

    verbose : boolean
        Print progress to the screen.

    version : integer
        Which version.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> dkt = desikan_killiany_tourville_labeling(image)
    """


    dkt = None
    t1 = ants.image_clone(t1, pixeltype='float')
    if version == 0:
        dkt = desikan_killiany_tourville_labeling_version0(t1,
                                                           do_preprocessing=do_preprocessing,
                                                           return_probability_images=return_probability_images,
                                                           do_lobar_parcellation=do_lobar_parcellation,
                                                           verbose=verbose
                                                           )
    elif version == 1:
        dkt = desikan_killiany_tourville_labeling_version1(t1,
                                                           do_preprocessing=do_preprocessing,
                                                           return_probability_images=return_probability_images,
                                                           do_lobar_parcellation=do_lobar_parcellation,
                                                           verbose=verbose
                                                           )
    return dkt


def desikan_killiany_tourville_labeling_version0(t1,
                                                 do_preprocessing=True,
                                                 return_probability_images=False,
                                                 do_lobar_parcellation=False,
                                                 verbose=False):

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import deep_atropos

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    template_transform_type = "antsRegistrationSyNQuickRepro[a]"
    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = ants.image_clone(t1)
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=(0.01, 0.99),
            brain_extraction_modality="t1",
            template="croppedMni152",
            template_transform_type=template_transform_type,
            do_bias_correction=True,
            do_denoising=True,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

    ################################
    #
    # Download spatial priors for outer model
    #
    ################################

    spatial_priors_file_name_path = get_antsxnet_data("priorDktLabels")
    spatial_priors = ants.image_read(spatial_priors_file_name_path)
    priors_image_list = ants.ndimage_to_list(spatial_priors)

    ################################
    #
    # Build outer model and load weights
    #
    ################################

    template_size = (96, 112, 96)
    labels = (0, 1002, 1003, *tuple(range(1005, 1032)), 1034, 1035,
                2002, 2003, *tuple(range(2005, 2032)), 2034, 2035)
    channel_size = 1 + len(priors_image_list)

    unet_model = create_unet_model_3d((*template_size, channel_size),
        number_of_outputs = len(labels),
        number_of_layers = 4, number_of_filters_at_base_layer = 16, dropout_rate = 0.0,
        convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
        weight_decay = 1e-5, additional_options=("attentionGating"))

    weights_file_name = None
    weights_file_name = get_pretrained_network("dktOuterWithSpatialPriors")
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Outer model Prediction.")

    downsampled_image = ants.resample_image(t1_preprocessed, template_size, use_voxels=True, interp_type=0)
    image_array = downsampled_image.numpy()
    image_array = (image_array - image_array.mean()) / image_array.std()

    batchX = np.zeros((1, *template_size, channel_size))
    batchX[0,:,:,:,0] = image_array

    for i in range(len(priors_image_list)):
        resampled_prior_image = ants.resample_image(priors_image_list[i], template_size, use_voxels=True, interp_type=0)
        batchX[0,:,:,:,i+1] = resampled_prior_image.numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    outer_probability_images = list()
    for i in range(len(labels)):
        probability_image = \
            ants.from_numpy_like(np.squeeze(predicted_data[0, :, :, :, i]), downsampled_image)
        resampled_image = ants.resample_image(probability_image, t1_preprocessed.shape, use_voxels=True, interp_type=0)
        if do_preprocessing:
            outer_probability_images.append(ants.apply_transforms(fixed=t1,
                moving=resampled_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", singleprecision=True,
                verbose=verbose))
        else:
            outer_probability_images.append(resampled_image)

    image_matrix = ants.image_list_to_matrix(outer_probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    dkt_label_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        dkt_label_image[segmentation_image==i] = labels[i]

    ################################
    #
    # Build inner model and load weights
    #
    ################################

    template_size = (160, 192, 160)
    labels = (0, 4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24,
                26, 28, 30, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 58,
                60, 91, 92, 630, 631, 632)

    unet_model = create_unet_model_3d((*template_size, 1),
        number_of_outputs = len(labels),
        number_of_layers = 4, number_of_filters_at_base_layer = 8, dropout_rate = 0.0,
        convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
        weight_decay = 1e-5, additional_options=("attentionGating"))

    weights_file_name = get_pretrained_network("dktInner")
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Prediction.")

    cropped_image = ants.crop_indices(t1_preprocessed, (12, 14, 0), (172, 206, 160))

    batchX = np.expand_dims(cropped_image.numpy(), axis=0)
    batchX = np.expand_dims(batchX, axis=-1)
    batchX = (batchX - batchX.mean()) / batchX.std()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    origin = cropped_image.origin
    spacing = cropped_image.spacing
    direction = cropped_image.direction

    inner_probability_images = list()
    for i in range(len(labels)):
        probability_image = \
            ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
            origin=origin, spacing=spacing, direction=direction)
        if i > 0:
            decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0)
        else:
            decropped_image = ants.decrop_image(probability_image, t1_preprocessed * 0 + 1)

        if do_preprocessing:
            inner_probability_images.append(ants.apply_transforms(fixed=t1,
                moving=decropped_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", singleprecision=True,
                verbose=verbose))
        else:
            inner_probability_images.append(decropped_image)

    image_matrix = ants.image_list_to_matrix(inner_probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    ################################
    #
    # Incorporate the inner model results into the final label image.
    # Note that we purposely prioritize the inner label results.
    #
    ################################

    for i in range(len(labels)):
        if labels[i] > 0:
            dkt_label_image[segmentation_image==i] = labels[i]

    if do_lobar_parcellation:

        if verbose:
            print("Doing lobar parcellation.")

        ################################
        #
        # Lobar/hemisphere parcellation
        #
        ################################

        # Consolidate lobar cortical labels

        if verbose:
            print("   Consolidating cortical labels.")

        frontal_labels = (1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1024, 1026, 1027, 1028,
                            2002, 2003, 2012, 2014, 2017, 2018, 2019, 2020, 2024, 2026, 2027, 2028)
        parietal_labels = (1008, 1010, 1022, 1023, 1025, 1029, 1031,
                            2008, 2010, 2022, 2023, 2025, 2029, 2031)
        temporal_labels = (1006, 1007, 1009, 1015, 1016, 1030, 1034,
                            2006, 2007, 2009, 2015, 2016, 2030, 2034)
        occipital_labels = (1005, 1011, 1013, 1021,
                            2005, 2011, 2013, 2021)

        lobar_labels = list()
        lobar_labels.append(frontal_labels)
        lobar_labels.append(parietal_labels)
        lobar_labels.append(temporal_labels)
        lobar_labels.append(occipital_labels)

        dkt_lobes = ants.image_clone(dkt_label_image)
        dkt_lobes[dkt_lobes < 1000] = 0

        for i in range(len(lobar_labels)):
            for j in range(len(lobar_labels[i])):
                dkt_lobes[dkt_lobes == lobar_labels[i][j]] = i + 1

        dkt_lobes[dkt_lobes > len(lobar_labels)] = 0

        six_tissue = deep_atropos(t1_preprocessed, do_preprocessing=False, verbose=verbose)
        atropos_seg = six_tissue['segmentation_image']
        if do_preprocessing:
            atropos_seg = ants.apply_transforms(fixed=t1, moving=atropos_seg,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="genericLabel", singleprecision=True,
                verbose=verbose)

        brain_mask = ants.image_clone(atropos_seg)
        brain_mask[brain_mask == 1] = 0
        brain_mask[brain_mask == 5] = 0
        brain_mask[brain_mask == 6] = 0
        brain_mask = ants.threshold_image(brain_mask, 0, 0, 0, 1)

        lobar_parcellation = ants.iMath(brain_mask, "PropagateLabelsThroughMask", brain_mask * dkt_lobes)

        lobar_parcellation[atropos_seg == 5] = 5
        lobar_parcellation[atropos_seg == 6] = 6

        # Do left/right

        if verbose:
            print("   Doing left/right hemispheres.")

        left_labels = (*tuple(range(4, 8)), *tuple(range(10, 14)), 17, 18, 25, 26, 28, 30, 91,
                    1002, 1003, *tuple(range(1005, 1032)), 1034, 1035)
        right_labels = (*tuple(range(43, 47)), *tuple(range(49, 55)), 57, 58, 60, 62, 92, 2002, 2003,
                        *tuple(range(2005, 2032)), 2034, 2035)

        hemisphere_labels = list()
        hemisphere_labels.append(left_labels)
        hemisphere_labels.append(right_labels)

        dkt_hemispheres = ants.image_clone(dkt_label_image)

        for i in range(len(hemisphere_labels)):
            for j in range(len(hemisphere_labels[i])):
                dkt_hemispheres[dkt_hemispheres == hemisphere_labels[i][j]] = i + 1

        dkt_hemispheres[dkt_hemispheres > 2] = 0

        atropos_brain_mask = ants.threshold_image(atropos_seg, 0, 0, 0, 1)
        hemisphere_parcellation = ants.iMath(atropos_brain_mask, "PropagateLabelsThroughMask",
                                             atropos_brain_mask * dkt_hemispheres)

        hemisphere_parcellation *= ants.threshold_image(lobar_parcellation, 0, 0, 0, 1)
        hemisphere_parcellation[hemisphere_parcellation == 1] = 0
        hemisphere_parcellation[hemisphere_parcellation == 2] = 1
        hemisphere_parcellation *= 6
        lobar_parcellation += hemisphere_parcellation

    if return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                        'lobar_parcellation' : lobar_parcellation,
                        'inner_probability_images' : inner_probability_images,
                        'outer_probability_images' : outer_probability_images }
        return(return_dict)
    elif return_probability_images and not do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                        'inner_probability_images' : inner_probability_images,
                        'outer_probability_images' : outer_probability_images }
        return(return_dict)
    elif not return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                        'lobar_parcellation' : lobar_parcellation }
        return(return_dict)
    else:
        return(dkt_label_image)


def desikan_killiany_tourville_labeling_version1(t1,
                                                 do_preprocessing=True,
                                                 return_probability_images=False,
                                                 do_lobar_parcellation=False,
                                                 verbose=False):

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import brain_extraction
    from ..utilities import deep_atropos

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    def reshape_image(image, crop_size, interp_type = "linear"):
        image_resampled = None
        if interp_type == "linear":
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=0)
        else:
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=1)
        image_cropped = ants.pad_or_crop_image_to_size(image_resampled, crop_size)
        return image_cropped

    which_template = "hcpyaT1Template"
    template_transform_type = "antsRegistrationSyNQuick[a]"
    template = ants.image_read(get_antsxnet_data(which_template))

    cropped_template_size = (160, 192, 160)

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = ants.image_clone(t1)
    if do_preprocessing:
        t1_preprocessing = preprocess_brain_image(t1,
            truncate_intensity=None,
            brain_extraction_modality="t1threetissue",
            template=which_template,
            template_transform_type=template_transform_type,
            do_bias_correction=True,
            do_denoising=False,
            verbose=verbose)
        t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']
        t1_preprocessed = reshape_image(t1_preprocessed, crop_size=cropped_template_size)

    ################################
    #
    # Build outer model and load weights
    #
    ################################

    dkt_lateral_labels = (0,)
    deep_atropos_labels = tuple(range(1, 7))
    dkt_left_labels = (1002, 1003, *tuple(range(1005, 1032)), 1034, 1035)
    dkt_right_labels = (2002, 2003, *tuple(range(2005, 2032)), 2034, 2035)

    labels = sorted((*dkt_lateral_labels, *deep_atropos_labels, *dkt_left_labels))

    channel_size = 1
    number_of_classification_labels = len(labels)

    unet_model_pre = create_unet_model_3d((*cropped_template_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        weight_decay=0.0)

    penultimate_layer = unet_model_pre.layers[-2].output

    output2 = Conv3D(filters=1,
                     kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

    unet_model = Model(inputs=unet_model_pre.input, outputs=[unet_model_pre.output, output2])

    unet_model.load_weights(get_pretrained_network("DesikanKillianyTourvilleOuter"))

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Model prediction.")

    batchX = np.zeros((2, *cropped_template_size, channel_size))
    batchX[0,:,:,:,0] = t1_preprocessed.iMath("Normalize").numpy()
    batchX[1,:,:,:,0] = np.flip(batchX[0,:,:,:,0], axis=0)

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    labels = sorted((*dkt_lateral_labels, *dkt_left_labels))
    probability_images = [None] * len(labels)

    dkt_labels = list()
    dkt_labels.append(dkt_lateral_labels)
    dkt_labels.append(dkt_left_labels)

    for b in range(2):
        for i in range(len(dkt_labels)):
            for j in range(len(dkt_labels[i])):
                label = dkt_labels[i][j]
                label_index = labels.index(label)
                if label == 0:
                    probability_array = np.squeeze(np.sum(predicted_data[0][b, :, :, :, :8], axis=3))
                else:
                    probability_array = np.squeeze(predicted_data[0][b, :, :, :, label_index + len(deep_atropos_labels)])
                if b == 1:
                    probability_array = np.flip(probability_array, axis=0)
                probability_image = ants.from_numpy_like(probability_array, t1_preprocessed)
                if do_preprocessing:
                    probability_image = ants.pad_or_crop_image_to_size(probability_image, template.shape)
                    probability_image = ants.apply_transforms(fixed=t1,
                        moving=probability_image,
                        transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                        whichtoinvert=[True], interpolator="linear", singleprecision=True, verbose=verbose)
                if b == 0:
                    probability_images[label_index] = probability_image
                else:
                    probability_images[label_index] = 0.5 * (probability_images[label_index] + probability_image)

    if verbose:
        print("Constructing foreground probability image.")

    foreground_probability_array = np.squeeze(0.5 * (predicted_data[1][0,:,:,:,:] +
                                                  np.flip(predicted_data[1][1,:,:,:,:], axis=0)))
    foreground_probability_image = ants.from_numpy_like(foreground_probability_array, t1_preprocessed)
    if do_preprocessing:
        foreground_probability_image = ants.pad_or_crop_image_to_size(foreground_probability_image, template.shape)
        foreground_probability_image = ants.apply_transforms(fixed=t1,
            moving=foreground_probability_image,
            transformlist=t1_preprocessing['template_transforms']['invtransforms'],
            whichtoinvert=[True], interpolator="linear", singleprecision=True,
            verbose=verbose)

    for i in range(len(dkt_labels)):
        for j in range(len(dkt_labels[i])):
            label = dkt_labels[i][j]
            label_index = labels.index(label)
            if label == 0:
                probability_images[label_index] *= (foreground_probability_image * -1 + 1)
            else:
                probability_images[label_index] *= foreground_probability_image

    labels = sorted((*dkt_lateral_labels, *dkt_left_labels))

    bext = brain_extraction(t1, modality="t1hemi", verbose=verbose)

    probability_all_images = list()
    probability_all_images.append(probability_images[0])
    for i in range(len(dkt_left_labels)):
        probability_image = ants.image_clone(probability_images[i+1])
        probability_left_image = probability_image * bext['probability_images'][1]
        probability_all_images.append(probability_left_image)
    for i in range(len(dkt_right_labels)):
        probability_image = ants.image_clone(probability_images[i+1])
        probability_right_image = probability_image * bext['probability_images'][2]
        probability_all_images.append(probability_right_image)

    image_matrix = ants.image_list_to_matrix(probability_all_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    dkt_all_labels = sorted((*dkt_lateral_labels, *dkt_left_labels, *dkt_right_labels))

    dkt_label_image = segmentation_image * 0
    for i in range(len(dkt_all_labels)):
        label = dkt_all_labels[i]
        label_index = dkt_all_labels.index(label)
        dkt_label_image[segmentation_image==label_index] = label

    if do_lobar_parcellation:

        if verbose:
            print("Doing lobar parcellation.")

        ################################
        #
        # Lobar/hemisphere parcellation
        #
        ################################

        # Consolidate lobar cortical labels

        if verbose:
            print("   Consolidating cortical labels.")

        frontal_labels = (1002, 1003, 1012, 1014, 1017, 1018, 1019, 1020, 1024, 1026, 1027, 1028,
                          2002, 2003, 2012, 2014, 2017, 2018, 2019, 2020, 2024, 2026, 2027, 2028)
        parietal_labels = (1008, 1010, 1022, 1023, 1025, 1029, 1031,
                           2008, 2010, 2022, 2023, 2025, 2029, 2031)
        temporal_labels = (1006, 1007, 1009, 1015, 1016, 1030, 1034,
                           2006, 2007, 2009, 2015, 2016, 2030, 2034)
        occipital_labels = (1005, 1011, 1013, 1021,
                            2005, 2011, 2013, 2021)

        lobar_labels = list()
        lobar_labels.append(frontal_labels)
        lobar_labels.append(parietal_labels)
        lobar_labels.append(temporal_labels)
        lobar_labels.append(occipital_labels)

        dkt_lobes = ants.image_clone(dkt_label_image)

        for i in range(len(lobar_labels)):
            for j in range(len(lobar_labels[i])):
                dkt_lobes[dkt_lobes == lobar_labels[i][j]] = i + 1

        dkt_lobes[dkt_lobes > len(lobar_labels)] = 0

        six_tissue = deep_atropos([t1, None, None], do_preprocessing=True, verbose=verbose)
        atropos_seg = six_tissue['segmentation_image']
        atropos_seg[atropos_seg == 1] = 0
        atropos_seg[atropos_seg == 5] = 0
        atropos_seg[atropos_seg == 6] = 0

        brain_mask = ants.image_clone(atropos_seg)
        brain_mask = ants.threshold_image(brain_mask, 0, 0, 0, 1)

        lobar_parcellation = ants.iMath(brain_mask, "PropagateLabelsThroughMask", brain_mask * dkt_lobes)

        # Do left/right

        if verbose:
            print("   Doing left/right hemispheres.")

        hemisphere_labels = list()
        hemisphere_labels.append(dkt_left_labels)
        hemisphere_labels.append(dkt_right_labels)

        dkt_hemispheres = ants.image_clone(dkt_label_image)

        for i in range(len(hemisphere_labels)):
            for j in range(len(hemisphere_labels[i])):
                dkt_hemispheres[dkt_hemispheres == hemisphere_labels[i][j]] = i + 1

        dkt_hemispheres[dkt_hemispheres > 2] = 0

        atropos_brain_mask = ants.threshold_image(atropos_seg, 0, 0, 0, 1)
        hemisphere_parcellation = ants.iMath(atropos_brain_mask, "PropagateLabelsThroughMask",
                                             atropos_brain_mask * dkt_hemispheres)

        hemisphere_parcellation *= ants.threshold_image(lobar_parcellation, 0, 0, 0, 1)
        hemisphere_parcellation[hemisphere_parcellation == 1] = 0
        hemisphere_parcellation[hemisphere_parcellation == 2] = 1
        hemisphere_parcellation *= 4
        lobar_parcellation += hemisphere_parcellation

    if return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'lobar_parcellation' : lobar_parcellation,
                       'probability_images' : probability_all_images }
        return(return_dict)
    elif return_probability_images and not do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'probability_images' : probability_all_images }
        return(return_dict)
    elif not return_probability_images and do_lobar_parcellation:
        return_dict = {'segmentation_image' : dkt_label_image,
                       'lobar_parcellation' : lobar_parcellation }
        return(return_dict)
    else:
        return(dkt_label_image)