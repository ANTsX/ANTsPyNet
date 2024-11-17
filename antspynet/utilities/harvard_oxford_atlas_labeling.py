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
    Cortical and deep gray matter labeling using Desikan-Killiany-Tourville
    
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

    return_probability_images : boolean
        Whether to return the two sets of probability images for the subcortical
        labels.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> dkt = harvard_oxford_atlas_labeling(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    def reshape_image(image, crop_size, interp_type = "linear"):
        image_resampled = None
        if interp_type == "linear":
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=0)
        else:        
            image_resampled = ants.resample_image(image, (1, 1, 1), use_voxels=False, interp_type=1)
        image_cropped = pad_or_crop_image_to_size(image_resampled, crop_size)
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
            brain_extraction_modality="bw20",
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

    labels = tuple(range(36))
    channel_size = 1
    number_of_classification_labels = len(labels)

    unet_model_pre = create_unet_model_3d((*cropped_template_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        weight_decay=0.0)

    penultimate_layer = unet_model_pre.layers[-2].output

    # medial temporal lobe
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
        print("Model Prediction.")

    batchX = np.zeros((1, *cropped_template_size, channel_size))
    batchX[0,:,:,:,0] = t1_preprocessed.iMath("Normalize").numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)[0]

    probability_images = list()
    hoa_labels = tuple(range(33))

    for i in range(len(hoa_labels)):
        probability_image = \
            ants.from_numpy_like(np.squeeze(predicted_data[0, :, :, :, i]), t1_preprocessed)
        if i == 0:
            probability_image += ants.from_numpy_like(np.squeeze(np.sum(predicted_data[0, :, :, :, 33:36], axis=3)), t1_preprocessed)
        if do_preprocessing:
            probability_image = pad_or_crop_image_to_size(probability_image, template.shape)
            probability_images.append(ants.apply_transforms(fixed=t1,
                moving=probability_image,
                transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                whichtoinvert=[True], interpolator="linear", verbose=verbose))
        else:
            probability_images.append(probability_image)

    image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    hoa_label_image = ants.image_clone(segmentation_image)
    for i in range(len(hoa_labels)):
        print("HOA label" + str(hoa_labels[i]))
        hoa_label_image[segmentation_image==i] = hoa_labels[i]

    return_dict = {'segmentation_image' : hoa_label_image,
                   'probability_images' : probability_images}
    return(return_dict)
