import numpy as np
import ants

import keras
from keras import layers, ops, Model

def cit168_labeling(t1, verbose=False):

    """
    
    Perform CIT168 segmentation in T1 images using Pauli atlas (CIT168) 
    labels described in https://pubmed.ncbi.nlm.nih.gov/29664465/

                group_labels = [0,7,8,9,23,24,25,33,34]

                            group_labels = [0,1,2,5,6,17,18,21,22]

    The labeling is as follows:

    Label 1:   BN_STR_Pu_Left
    Labeļ 2:   BN_STR_Ca_Left
    Label 3:   BN_STR_NAC_Left.  (not modeled)
    Label 4:   EXA_Left          (not modeled)
    Label 5:   BN_GP_GPe_Left
    Label 6:   BN_GP_GPi_Left 
    Label 7:   MTg_SN_SNc_Left
    Label 8:   MTg_RN_Left
    Label 9:   MTg_SN_SNc_Left
    Label 10:  MTg_VTR_PBP_Left  (not modeled)
    Label 11:  MTg_VTR_VTA_Left  (not modeled)
    Label 12:  BN_GP_VeP_Left    (not modeled)
    Label 13:  THM_ETH_HN_Left   (not modeled)
    Label 14:  Die_HTH_Left      (not modeled)
    Label 15:  Die_HTH_MN_Left   (not modeled) 
    Label 16:  Die_STH_Left      (not modeled)
    Label 17:  BN_STR_Pu_Right
    Label 18:  BN_STR_Ca_Right
    Label 19:  BN_STR_NAC_Right  (not modeled)
    Label 20:  EXA_Right         (not modeled)
    Label 21:  BN_GP_GPe_Right
    Label 22:  BN_GP_GPi_Right
    Label 23:  MTg_SN_SNc_Right
    Label 24:  MTg_RN_Right
    Label 25:  MTg_SN_SNr_Right
    Label 26:  MTg_VTR_PBP_Right (not modeled)
    Label 27:  MTg_VTR_VTA_Right (not modeled)
    Label 28:  BN_GP_VeP_Right   (not modeled)
    Label 29:  THM_ETH_HN_Right  (not modeled)
    Label 30:  Die_HTH_Right     (not modeled)
    Label 31:  Die_HTH_MN_Right  (not modeled)
    Label 32:  Die_STH_Right     (not modeled)
    Label 33:  ReferenceRegion_Left
    Label 34:  ReferenceRegion_Right

    Preprocessing consists of:
       * n4 bias correction and
       * brain extraction
    
    Arguments
    ---------
    t1 : ANTsImage
        input image

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    ANTs segmentation label and probability images.

    Example
    -------
    >>> seg = cit168_labeling(t1)
    """

    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities import get_antsxnet_data
    from ..architectures import create_unet_model_3d
    
    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    template = ants.image_read(get_antsxnet_data("CIT168_T1w_700um_pad_adni"))
    template_small = ants.resample_image(template, [2, 2, 2])
    template_large = ants.resample_image(template, [0.5, 0.5, 0.5])
    template_seg = ants.image_read(get_antsxnet_data("det_atlas_25_pad_LR_adni"))

    template_transform_type = "antsRegistrationSyNQuickRepro[s]"

    cropped_template_size = [160, 160, 112]

    ################################
    #
    # Preprocess images
    #
    ################################

    # clone input to float
    t1 = ants.image_clone(t1, pixeltype="float")

    t1_preprocessing = preprocess_brain_image(t1,
        truncate_intensity=[1e-4, 0.999],
        brain_extraction_modality="t1threetissue",
        template=None,
        do_bias_correction=True,
        do_denoising=False,
        intensity_normalization_type='01',
        verbose=verbose)
                        
    t1_preprocessed = t1_preprocessing['preprocessed_image'] * t1_preprocessing['brain_mask']

    reg = ants.registration(template_small, t1_preprocessed, type_of_transform=template_transform_type, verbose=verbose)  
    t1_preprocessed = ants.apply_transforms(fixed=template_large, moving=t1_preprocessed,
                                            transformlist=reg['fwdtransforms'][1],
                                            interpolator="linear", singleprecision=True, verbose=verbose)

    template_priors = ants.apply_transforms(fixed=t1_preprocessed, moving=template_seg,
                                            transformlist=reg['invtransforms'][1],
                                            interpolator="nearestNeighbor", singleprecision=True, verbose=verbose)
                                            
    center_of_mass = list(ants.get_center_of_mass(ants.threshold_image(template_priors, 0, 0, 0, 1)))
    center_of_mass[1] = center_of_mass[1] + 10.0  # same as we did in training
    t1_cropped = ants.crop_image_from_center_point(t1_preprocessed, center_of_mass, cropped_template_size)
    template_priors_cropped = ants.crop_image_from_center_point(template_priors, center_of_mass, cropped_template_size)
     
    ################################
    #
    # Build model and load weights
    #
    ################################


    cit168_segmentation_image = t1 * 0
    for sn in [True,False]:

        if verbose:
            print("    CIT168: Creating model and loading weights. (sn=" + str(sn) + ")")

        if sn:
            group_labels = [0,7,8,9,23,24,25,33,34]
            cit168_weights = get_pretrained_network("deepCIT168_sn")
        else:
            group_labels = [0,1,2,5,6,17,18,21,22]
            cit168_weights = get_pretrained_network("deepCIT168")

        channel_size = 1
        number_of_outputs = len(group_labels)
        number_of_channels = len(group_labels)

        unet0 = create_unet_model_3d([None, None, None, number_of_channels],
                                     number_of_outputs=1, number_of_layers=4,
                                     number_of_filters_at_base_layer=32, 
                                     convolution_kernel_size=3, 
                                     deconvolution_kernel_size=2,
                                     pool_size=2, strides=2, dropout_rate=0.0,
                                     weight_decay=0,
                                     additional_options="nnUnetActivationStyle",
                                     mode="sigmoid")
        
        unet1 = create_unet_model_3d([None, None, None, 2],
                                     number_of_outputs=number_of_outputs,
                                     number_of_filters=(32, 64, 96, 128, 256),
                                     convolution_kernel_size=(3, 3, 3),
                                     deconvolution_kernel_size=(2, 2, 2),
                                     dropout_rate=0.0, weight_decay=0,
                                     additional_options = "nnUnetActivationStyle",
                                     mode="classification")

        splits = ops.split(unet0.inputs[0], 9, axis=-1)
        newmult = layers.Concatenate(axis=-1)([splits[0], unet0.outputs[0]])
        unetonnet = unet1(newmult)
        unet_model = Model(inputs=unet0.inputs, outputs=[unetonnet, unet0.outputs[0]])

        unet_model.load_weights(cit168_weights)

        priors = ants.segmentation_to_one_hot(template_priors_cropped.numpy().astype('int'), 
                                              segmentation_labels=group_labels[1:len(group_labels)])
        batchX = np.concatenate((np.expand_dims(t1_cropped.numpy(), axis=-1), priors), axis=-1)

        if verbose:
            print("    CIT168: Model prediction.")
       
        predicted_data = unet_model.predict(np.expand_dims(batchX, axis=0), verbose=verbose)

        probability_images = []
        for i in range(1, len(group_labels)):
            probability_array = np.squeeze(predicted_data[0][0,:,:,:,i])
            probability_image = ants.from_numpy_like(probability_array, t1_cropped)
            probability_image = ants.apply_transforms(fixed=t1, moving=probability_image,
                 transformlist=reg['invtransforms'][0], whichtoinvert=[True], 
                 interpolator="linear", singleprecision=True, verbose=verbose)
            probability_images.append(probability_image)
   
        predicted_mask = ants.from_numpy_like(np.squeeze(predicted_data[1][0,:,:,:,0]), t1_cropped) 
        predicted_mask = ants.apply_transforms(fixed=t1, moving=predicted_mask,
                 transformlist=reg['invtransforms'][0], whichtoinvert=[True], 
                 interpolator="linear", singleprecision=True, verbose=verbose)
        predicted_mask = ants.threshold_image(predicted_mask, 0.5, 1, 1, 0)

        image_matrix = ants.image_list_to_matrix(probability_images, predicted_mask)
        segmentation_matrix = np.argmax(image_matrix, axis=0) + 1
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), predicted_mask)[0]
        relabeled_segmentation_image = segmentation_image * 0
        for i in range(1, len(group_labels)):
            single_label_image = ants.threshold_image(segmentation_image, i, i, 1, 0)
            if group_labels[i] < 33:
                single_label_image = ants.iMath_get_largest_component(single_label_image, 1)
            relabeled_segmentation_image += single_label_image * group_labels[i]

        cit168_segmentation_image += relabeled_segmentation_image
    
    return cit168_segmentation_image

