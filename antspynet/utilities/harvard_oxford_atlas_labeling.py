import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import ants

def harvard_oxford_atlas_labeling(t1,
                                  do_preprocessing=True,
                                  verbose=False):

    """
    Subcortical and cerebellar labeling from a T1 image.

    Perform HOA labeling using deep learning and data from "High Resolution,
    Comprehensive Atlases of the Human Brain Morphology" number: "NIH NIMH
    5R01MH112748-04". Repository: 'https://github.com/HOA-2/SubcorticalParcellations'

    The labeling is as follows:

    Label 1: 	Lateral Ventricle Left
    Label 2:	Lateral Ventricle Right
    Label 3:	CSF
    Label 4:	Third Ventricle
    Label 5:	Fourth Ventricle
    Label 6:	5th Ventricle
    Label 7:	Nucleus Accumbens Left
    Label 8:	Nucleus Accumbens Right
    Label 9:	Caudate Left
    Label 10:	Caudate Right
    Label 11:	Putamen Left
    Label 12:	Putamen Right
    Label 13:	Globus Pallidus Left
    Label 14:	Globus Pallidus Right
    Label 15:	Brainstem
    Label 16:	Thalamus Left
    Label 17:	Thalamus Right
    Label 18:	Inferior Horn of the Lateral Ventricle Left
    Label 19:	Inferior Horn of the Lateral Ventricle Right
    Label 20:	Hippocampal Formation Left
    Label 21:	Hippocampal Formation Right
    Label 22:	Amygdala Left
    Label 23:	Amygdala Right
    Label 24:	Optic Chiasm
    Label 25:	VDC Anterior Left
    Label 26:	VDC Anterior Right
    Label 27:	VDC Posterior Left
    Label 28:	VDC Posterior Right
    Label 29:	Cerebellar Cortex Left
    Label 30:	Cerebellar Cortex Right
    Label 31:	Cerebellar White Matter Left
    Label 32:	Cerebellar White Matter Right

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

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Dict consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> hoa = harvard_oxford_atlas_labeling(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import brain_extraction

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    # clone input to float
    t1 = ants.image_clone(t1, pixeltype="float")

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

    cropped_template_size = (160, 176, 160)

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
    # Build model and load weights
    #
    ################################

    hoa_lateral_labels = (0, 3, 4, 5, 6, 15, 24)
    hoa_lateral_left_labels = (1, 7, 9, 11, 13, 16, 18, 20, 22, 25, 27, 29, 31)
    hoa_lateral_right_labels = (2, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28, 30, 32)
    hoa_extra_labels = (33, 34, 35)

    labels = sorted((*hoa_lateral_labels, *hoa_lateral_left_labels, *hoa_extra_labels))

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

    unet_model.load_weights(get_pretrained_network("HarvardOxfordAtlasSubcortical"))

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Model prediction using both the original and contralaterally flipped version.")

    batchX = np.zeros((2, *cropped_template_size, channel_size))
    batchX[0,:,:,:,0] = t1_preprocessed.iMath("Normalize").numpy()
    batchX[1,:,:,:,0] = np.flip(batchX[0,:,:,:,0], axis=0)

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    labels = sorted((*hoa_lateral_labels, *hoa_lateral_left_labels))
    probability_images = [None] * len(labels)

    hoa_labels = list()
    hoa_labels.append(hoa_lateral_labels)
    hoa_labels.append(hoa_lateral_left_labels)
    # hoa_labels.append(hoa_lateral_right_labels)

    for b in range(2):
        for i in range(len(hoa_labels)):
            for j in range(len(hoa_labels[i])):
                label = hoa_labels[i][j]
                label_index = labels.index(label)
                probability_array = np.squeeze(predicted_data[0][b, :, :, :, label_index])
                if label == 0:
                    probability_array += np.squeeze(np.sum(predicted_data[0][b, :, :, :, 20:], axis=3))
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
            whichtoinvert=[True], interpolator="linear", singleprecision=True, verbose=verbose)

    for i in range(len(hoa_labels)):
        for j in range(len(hoa_labels[i])):
            label = hoa_labels[i][j]
            label_index = labels.index(label)
            if label == 0:
                probability_images[label_index] *= (foreground_probability_image * -1 + 1)
            else:
                probability_images[label_index] *= foreground_probability_image

    labels = sorted((*hoa_lateral_labels, *hoa_lateral_left_labels))

    bext = brain_extraction(t1, modality="t1hemi", verbose=verbose)

    probability_all_images = list()
    for i in range(len(labels)):
        probability_image = ants.image_clone(probability_images[i])
        if labels[i] in hoa_lateral_left_labels:
            probability_left_image = probability_image * bext['probability_images'][1]
            probability_right_image = probability_image * bext['probability_images'][2]
            probability_all_images.append(probability_left_image)
            probability_all_images.append(probability_right_image)
        else:
            probability_all_images.append(probability_image)

    image_matrix = ants.image_list_to_matrix(probability_all_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    hoa_all_labels = sorted((*hoa_lateral_labels, *hoa_lateral_left_labels, *hoa_lateral_right_labels))

    hoa_label_image = segmentation_image * 0
    for i in range(len(hoa_all_labels)):
        label = hoa_all_labels[i]
        label_index = hoa_all_labels.index(label)
        hoa_label_image[segmentation_image==label_index] = label

    return_dict = {'segmentation_image' : hoa_label_image,
                   'probability_images' : probability_all_images}
    return(return_dict)
