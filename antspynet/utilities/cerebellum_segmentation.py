import numpy as np
import ants

from tensorflow.keras.layers import Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

def cerebellum_segmentation(t1,
                            initial_cerebellum_mask=None,
                            do_preprocessing=True,
                            antsxnet_cache_directory=None,
                            verbose=False
                            ):

    """
    Cerebellum tissue segmentation and Schmahmann parcellation.

    Perform cerebellum segmentation using a modification of the set of 
    MaGET cerebellum atlases --- https://github.com/CoBrALab/MAGeTbrain.

    The tissue labeling is as follows:
    
    Label 1 : CSF
    Label 2 : Gray matter
    Label 3 : White matter
    
    The parcellation labeling is as follows:

    Label 1   : L_I_II
    Label 2   : L_III
    Label 3   : L_IV
    Label 4   : L_V
    Label 5   : L_VI
    Label 6   : L_Crus_I
    Label 7   : L_Crus_II
    Label 8   : L_VIIB
    Label 9   : L_VIIIA
    Label 10  : L_VIIIB
    Label 11  : L_IX
    Label 12  : L_X
    Label 13  : L_CM
    
    Label 100 : CSF
    
    Label 101 : R_I_II
    Label 102 : R_III
    Label 103 : R_IV
    Label 104 : R_V
    Label 105 : R_VI
    Label 106 : R_Crus_I
    Label 107 : R_Crus_II
    Label 108 : R_VIIB
    Label 109 : R_VIIIA
    Label 110 : R_VIIIB
    Label 111 : R_IX
    Label 112 : R_X
    Label 113 : R_CM

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * affine registration to the "deep flash" template.
    which is performed on the input images if do_preprocessing = True.

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    initial_cerebellum_mask : ANTsImage
        First option for initialization.  If not specified, and if the brain_mask
        is not specified, the cerebellum ROI is determined using ANTsXNet 
        brain_extraction followed by registration to a template.

    brain_mask : ANTsImage
        Second option for initialization. 
        
    do_preprocessing : boolean
        See description above.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the multiple segmentation images and probability images for
    each label and foreground.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> seg = cerebellum_segmentation(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import brain_extraction

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    transform_type = "antsRegistrationSyNQuick[s]" 
    whichtoinvert=[True, False, True]
    # transform_type = "antsRegistrationSyNQuick[a]" 
    # whichtoinvert=[True, True]

    ################################
    #
    # Get the templates, masks, priors
    #
    ################################

    t1_template = ants.image_read(get_antsxnet_data("magetTemplate"))
    t1_template_brain_mask = ants.image_read(get_antsxnet_data("magetTemplateBrainMask"))
    t1_template_brain = t1_template * t1_template_brain_mask
    t1_cerebellum_template = ants.image_read(get_antsxnet_data("magetCerebellumTemplate"))
    cerebellum_x_template_xfrm = get_antsxnet_data("magetCerebellumxTemplate0GenericAffine")

    # spatial priors are in the space of the cerebellar template.  First three are 
    # csf, gm, and wm followed by the regions.  
    spatial_priors_file_name_path = get_antsxnet_data("magetCerebellumTemplatePriors",
        antsxnet_cache_directory=antsxnet_cache_directory)
    spatial_priors = ants.image_read(spatial_priors_file_name_path)
    priors_image_list = ants.ndimage_to_list(spatial_priors)
    for i in range(len(priors_image_list)):
        priors_image_list[i] = ants.copy_image_info(t1_cerebellum_template, priors_image_list[i])

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = t1
    t1_mask = None

    template_transforms = None
    if do_preprocessing:
        
        if verbose:
            print("Preprocessing T1.")

        # Do bias correction
        t1_preprocessed = ants.n4_bias_field_correction(t1_preprocessed, t1_mask, shrink_factor=4, verbose=verbose)

    if initial_cerebellum_mask is None:
        # Brain extraction
        probability_mask = brain_extraction(t1_preprocessed, modality="t1",
            antsxnet_cache_directory=antsxnet_cache_directory, verbose=verbose)
        t1_mask = ants.threshold_image(probability_mask, 0.5, 1, 1, 0)
        t1_brain_preprocessed = t1_preprocessed * t1_mask

        # Warp to template and concatenate with cerebellum x template transform
        if verbose:
            print("Register T1 to whole brain template.")
            
        registration = ants.registration(fixed=t1_template_brain, moving=t1_brain_preprocessed,
            type_of_transform=transform_type, verbose=verbose)
        registration['invtransforms'].append(cerebellum_x_template_xfrm)
        registration['fwdtransforms'].insert(0, cerebellum_x_template_xfrm)
        template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                                   invtransforms=registration['invtransforms'])
    else:
        t1_cerebellum_template_mask = ants.threshold_image(t1_cerebellum_template, -0.01, 100, 0, 1)
        t1_cerebellum_template_mask = ants.apply_transforms(t1_template, t1_cerebellum_template_mask, 
                                                            transformlist=cerebellum_x_template_xfrm,
                                                            interpolator='nearestNeighbor',
                                                            whichtoinvert=[True])
        ants.image_write(t1_cerebellum_template_mask, "~/Desktop/t1_cerebellum_template_mask.nii.gz")    

        if verbose:
            print("Register T1 cerebellum to the cerebellum of the whole brain template.")
            
        registration = ants.registration(fixed=t1_template_brain * t1_cerebellum_template_mask, 
                                         moving=t1_preprocessed * initial_cerebellum_mask,
                                         type_of_transform=transform_type, verbose=verbose)
        registration['invtransforms'].append(cerebellum_x_template_xfrm)
        registration['fwdtransforms'].insert(0, cerebellum_x_template_xfrm)
        template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                                   invtransforms=registration['invtransforms'])
        

    t1_preprocessed_in_cerebellum_space = ants.apply_transforms(t1_cerebellum_template, t1_preprocessed,
                                                                transformlist=registration['fwdtransforms'])

    ants.image_write(t1_preprocessed_in_cerebellum_space, "~/Desktop/t1_preprocessed_in_cerebellum_space.nii.gz")    
    
    ################################
    #
    # Create model
    #
    ################################

    tissue_labels = (0, 1, 2, 3)
    region_labels = (0, *list(range(1, 14)), *list(range(100, 114)))

    image_size = (240, 144, 144)
    channel_size = 1 + len(priors_image_list)
    number_of_classification_labels = len(region_labels)

    unet_model = create_unet_model_3d((*image_size, channel_size),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(32, 64, 96, 128, 256),
        convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0, weight_decay=0)

    penultimate_layer = unet_model.layers[-2].output

    # Whole cerebellum 
    output2 = Conv3D(filters=1,
                     kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

    # CSF/GM/WM
    output3 = Conv3D(filters=4,
                     kernel_size=(1, 1, 1),
                     activation='sigmoid',
                     kernel_regularizer=regularizers.l2(0.0))(penultimate_layer)

    unet_model = Model(inputs=unet_model.input, outputs=[unet_model.output, output2, output3])

    ################################
    #
    # Load weights
    #
    ################################

    if verbose:
        print("Retrieving model weights.")
        
    network_name = "cerebellumHierarchical"
    weights_file_name = get_pretrained_network(network_name, antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Do prediction and normalize to native space
    #
    ################################

    if verbose:
        print("Prediction.")

    t1_preprocessed_in_cerebellum_space = \
        ((t1_preprocessed_in_cerebellum_space - t1_preprocessed_in_cerebellum_space.min()) /
         (t1_preprocessed_in_cerebellum_space.max() - t1_preprocessed_in_cerebellum_space.min()))

    batchX = np.zeros((2, *image_size, channel_size))
    batchX[0,:,:,:,0] = pad_or_crop_image_to_size(t1_preprocessed_in_cerebellum_space, image_size).numpy()
    batchX[1,:,:,:,0] = np.flip(batchX[0,:,:,:,0], axis=0)
    
    # padded = pad_or_crop_image_to_size(t1_preprocessed_in_cerebellum_space, image_size)
    # ants.image_write(ants.from_numpy(np.squeeze(batchX[0,:,:,:,0]), origin=padded.origin, spacing=padded.spacing, direction=padded.direction), "~/Desktop/BatchX0.nii.gz")
    # ants.image_write(ants.from_numpy(np.squeeze(batchX[1,:,:,:,0]), origin=padded.origin, spacing=padded.spacing, direction=padded.direction), "~/Desktop/BatchX1.nii.gz")

    for i in range(len(priors_image_list)):
        for j in range(batchX.shape[0]):
            batchX[j,:,:,:,i+1] = pad_or_crop_image_to_size(priors_image_list[i], image_size).numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)
    
    # region labels

    # first switch the contralateral labels for the flipped version 
    for i in range(1, 14):
        tmp_array = predicted_data[0][1,:,:,:,i]
        predicted_data[0][1,:,:,:,i] = predicted_data[0][1,:,:,:,i+14]
        predicted_data[0][1,:,:,:,i+14] = tmp_array

    region_probability_images = list()
    for i in range(len(region_labels)):
        probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[0][0,:,:,:,i]) + 
                                            np.flip(np.squeeze(predicted_data[0][1,:,:,:,i]), axis=0)))
        # probability_image = ants.from_numpy(np.squeeze(predicted_data[0][0,:,:,:,i]))
        probability_image = pad_or_crop_image_to_size(probability_image, t1_cerebellum_template.shape)
        probability_image = ants.copy_image_info(t1_cerebellum_template, probability_image)
        probability_image = ants.apply_transforms(fixed=t1,
            moving=probability_image,
            transformlist=template_transforms['invtransforms'],
            whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
        region_probability_images.append(probability_image)

    # whole cerebellum foreground
    probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[1][0,:,:,:,0]) + 
                                        np.flip(np.squeeze(predicted_data[1][1,:,:,:,0]), axis=0)))
    # probability_image = ants.from_numpy(np.squeeze(predicted_data[1][0,:,:,:,0]))
    probability_image = pad_or_crop_image_to_size(probability_image, t1_cerebellum_template.shape)
    probability_image = ants.copy_image_info(t1_cerebellum_template, probability_image)
    probability_image = ants.apply_transforms(fixed=t1,
        moving=probability_image,
        transformlist=template_transforms['invtransforms'],
        whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
    cerebellum_probability_image = probability_image

    # tissue labels
    tissue_probability_images = list()
    for i in range(len(tissue_labels)):
        probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[2][0,:,:,:,i]) + 
                                            np.flip(np.squeeze(predicted_data[2][1,:,:,:,i]), axis=0)))
        # probability_image = ants.from_numpy(np.squeeze(predicted_data[2][0,:,:,:,i]))
        probability_image = pad_or_crop_image_to_size(probability_image, t1_cerebellum_template.shape)
        probability_image = ants.copy_image_info(t1_cerebellum_template, probability_image)
        probability_image = ants.apply_transforms(fixed=t1,
            moving=probability_image,
            transformlist=template_transforms['invtransforms'],
            whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
        tissue_probability_images.append(probability_image)


    ################################
    #
    # Convert probability images to segmentations
    #
    ################################

    # region labels

    probability_images = region_probability_images
    labels = region_labels
    
    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    region_segmentation = ants.image_clone(relabeled_image)

    # tissue labels

    probability_images = tissue_probability_images
    labels = tissue_labels
    
    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], t1 * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], t1 * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    relabeled_image = ants.image_clone(segmentation_image)
    for i in range(len(labels)):
        relabeled_image[segmentation_image==i] = labels[i]

    tissue_segmentation = ants.image_clone(relabeled_image)

    return_dict = {'cerebellum_probability_image' : cerebellum_probability_image,
                   'parcellation_segmentation_image' : region_segmentation,
                   'parcellation_probability_images' : region_probability_images,
                   'tissue_segmentation_image' : tissue_segmentation,
                   'tissue_probability_images' : tissue_probability_images
                    }

    return(return_dict)


