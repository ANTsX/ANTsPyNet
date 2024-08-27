import numpy as np
import ants

def cerebellum_morphology(t1,
                          cerebellum_mask=None,
                          compute_thickness_image=False,
                          do_preprocessing=True,
                          verbose=False
                          ):

    """
    Cerebellum tissue segmentation, Schmahmann parcellation, and thickness.

    Perform cerebellum segmentation using a modification of the set of
    MaGET cerebellum atlases --- https://www.cobralab.ca/cerebellum-lobules

    https://www.nature.com/articles/s41598-024-59440-6

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

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted whole head ( not brain extracted ) image.

    cerebellum_mask : ANTsImage
        Option for initialization.  If not specified, the cerebellum ROI is
        determined using ANTsXNet brain_extraction followed by registration
        to a template.

    compute_thickness_image : boolean
        Compute KellyKapowski thickness image of the gray matter.

    do_preprocessing : boolean
        Perform N4 bias correction and spatiall normalize to template space.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the multiple segmentation images and probability images for
    each label and foreground.  Optional thickness image.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> cereb = cerebellum_morphology(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import brain_extraction

    if t1.dimension != 3:
        raise ValueError("Image dimension must be 3.")

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
    t1_cerebellum_template = ((t1_cerebellum_template - t1_cerebellum_template.min()) /
                              (t1_cerebellum_template.max() - t1_cerebellum_template.min()))
    cerebellum_x_template_xfrm = get_antsxnet_data("magetCerebellumxTemplate0GenericAffine")

    # spatial priors are in the space of the cerebellar template.  First three are
    # csf, gm, and wm followed by the regions.
    spatial_priors_file_name_path = get_antsxnet_data("magetCerebellumTemplatePriors")
    spatial_priors = ants.image_read(spatial_priors_file_name_path)
    priors_image_list = ants.ndimage_to_list(spatial_priors)
    for i in range(len(priors_image_list)):
        priors_image_list[i] = ants.copy_image_info(t1_cerebellum_template, priors_image_list[i])

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = ants.image_clone(t1)
    t1_mask = None

    template_transforms = None
    if do_preprocessing:

        if verbose:
            print("Preprocessing T1.")

        # Do bias correction
        t1_preprocessed = ants.n4_bias_field_correction(t1_preprocessed, shrink_factor=4, verbose=verbose)

    if cerebellum_mask is None:
        # Brain extraction
        probability_mask = brain_extraction(t1_preprocessed, modality="t1", verbose=verbose)
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
        t1_cerebellum_template_mask = ants.threshold_image(t1_cerebellum_template, 0.1, 1.1, 1, 0)
        t1_cerebellum_template_mask = ants.apply_transforms(t1_template, t1_cerebellum_template_mask,
                                                            transformlist=cerebellum_x_template_xfrm,
                                                            interpolator='nearestNeighbor',
                                                            whichtoinvert=[True])

        if verbose:
            print("Register T1 cerebellum to the cerebellum of the whole brain template.")

        registration = ants.registration(fixed=t1_template_brain * t1_cerebellum_template_mask,
                                         moving=t1_preprocessed * cerebellum_mask,
                                         type_of_transform=transform_type, verbose=verbose)
        registration['invtransforms'].append(cerebellum_x_template_xfrm)
        registration['fwdtransforms'].insert(0, cerebellum_x_template_xfrm)
        template_transforms = dict(fwdtransforms=registration['fwdtransforms'],
                                   invtransforms=registration['invtransforms'])

    t1_preprocessed_in_cerebellum_space = ants.apply_transforms(t1_cerebellum_template, t1_preprocessed,
                                                                transformlist=registration['fwdtransforms'])
    t1_preprocessed_mask_in_cerebellum_space = None
    if cerebellum_mask is not None:
        t1_preprocessed_mask_in_cerebellum_space = ants.apply_transforms(t1_cerebellum_template, cerebellum_mask,
                                                                         transformlist=registration['fwdtransforms'])


    ################################
    #
    # Create models, do prediction, and normalize to original t1 space
    #
    ################################

    tissue_labels = (0, 1, 2, 3)
    region_labels = (0, *list(range(1, 13)), *list(range(101, 113)))

    image_size = (240, 144, 144)

    cerebellum_probability_image = None
    tissue_probability_images = list()
    region_probability_images = list()
    which_priors = None

    start_m = 0
    if cerebellum_mask is not None:
        start_m = 1
        cerebellum_probability_image = ants.image_clone(cerebellum_mask)
    for m in range(start_m, 3):
        if m == 0:
            labels = (0, 1)
            channel_size = 2
            which_priors = None
            network_name = "cerebellumWhole"
            additional_options = ["attentionGating"]
        elif m == 1:
            labels = tissue_labels
            channel_size = len(labels)
            which_priors = (0, 1, 2)
            network_name = "cerebellumTissue"
            additional_options = None
        else:  #  m == 2:
            labels = region_labels
            channel_size = len(labels)
            which_priors = (*list(range(3, 15)), *list(range(16, 28)))
            network_name = "cerebellumLabels"
            additional_options = ["attentionGating"]

        number_of_classification_labels = len(labels)
        unet_model = create_unet_model_3d((*image_size, channel_size),
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_filters=(32, 64, 96, 128, 256),
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            dropout_rate=0.0, weight_decay=0, additional_options=additional_options)

        if verbose:
            print("Processing " + network_name)

        ################################
        #
        # Load weights
        #
        ################################

        if verbose:
            print("Retrieving model weights.")

        weights_file_name = get_pretrained_network(network_name)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction.")

        if m > 0:
            t1_preprocessed_in_cerebellum_space *= t1_preprocessed_mask_in_cerebellum_space

        t1_preprocessed_in_cerebellum_space = \
            ((t1_preprocessed_in_cerebellum_space - t1_preprocessed_in_cerebellum_space.min()) /
             (t1_preprocessed_in_cerebellum_space.max() - t1_preprocessed_in_cerebellum_space.min()))

        batchX = np.zeros((2, *image_size, channel_size))
        batchX[0,:,:,:,0] = pad_or_crop_image_to_size(t1_preprocessed_in_cerebellum_space, image_size).numpy()
        batchX[1,:,:,:,0] = np.flip(batchX[0,:,:,:,0], axis=0)

        if m == 0:
            for j in range(batchX.shape[0]):
                batchX[j,:,:,:,1] = pad_or_crop_image_to_size(t1_cerebellum_template, image_size).numpy()
        if m > 0:
            for i in range(len(which_priors)):
                for j in range(batchX.shape[0]):
                    batchX[j,:,:,:,i+1] = pad_or_crop_image_to_size(priors_image_list[which_priors[i]], image_size).numpy()

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        def decrop_to_cerebellum_template_space(target_image, reference_image):
            target_image_padded = pad_or_crop_image_to_size(target_image, reference_image.shape)
            one_padding_shape = np.array(target_image_padded.shape) + 1
            target_image_padded = ants.pad_image(target_image_padded, shape=one_padding_shape)
            lower_indices = np.array((1, 1, 0))
            upper_indices = np.array(reference_image.shape) + np.array((1, 1, 0))
            target_image_decropped = ants.crop_indices(target_image_padded, lower_indices, upper_indices)
            target_image_decropped = ants.copy_image_info(reference_image, target_image_decropped)
            return target_image_decropped

        if m == 0:

            # whole cerebellum foreground
            probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[0,:,:,:,1]) +
                                                np.flip(np.squeeze(predicted_data[1,:,:,:,1]), axis=0)))
            probability_image = decrop_to_cerebellum_template_space(probability_image, t1_cerebellum_template)
            t1_preprocessed_mask_in_cerebellum_space = ants.threshold_image(probability_image, 0.5, 1, 1, 0)
            probability_image = ants.apply_transforms(fixed=t1,
                moving=probability_image,
                transformlist=template_transforms['invtransforms'],
                whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
            cerebellum_probability_image = probability_image

        elif m == 1:

            # tissue labels
            for i in range(len(tissue_labels)):
                probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[0,:,:,:,i]) +
                                                    np.flip(np.squeeze(predicted_data[1,:,:,:,i]), axis=0)))
                probability_image = decrop_to_cerebellum_template_space(probability_image, t1_cerebellum_template)
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
                tissue_probability_images.append(probability_image)

        else:
            # region labels

            for i in range(1, 13):
                tmp_array = np.copy(predicted_data[1,:,:,:,i])
                predicted_data[1,:,:,:,i] = predicted_data[1,:,:,:,i+12]
                predicted_data[1,:,:,:,i+12] = np.copy(tmp_array)

            for i in range(len(region_labels)):
                probability_image = ants.from_numpy(0.5 * (np.squeeze(predicted_data[0,:,:,:,i]) +
                                                    np.flip(np.squeeze(predicted_data[1,:,:,:,i]), axis=0)))
                probability_image = decrop_to_cerebellum_template_space(probability_image, t1_cerebellum_template)
                probability_image = ants.apply_transforms(fixed=t1,
                    moving=probability_image,
                    transformlist=template_transforms['invtransforms'],
                    whichtoinvert=whichtoinvert, interpolator="linear", verbose=verbose)
                region_probability_images.append(probability_image)

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

    if compute_thickness_image:

        ################################
        #
        # Compute thickness image using KK
        #
        ################################

        kk = ants.kelly_kapowski(s=tissue_segmentation,
                                 g=tissue_probability_images[2],
                                 w=tissue_probability_images[3],
                                 its=45,
                                 r=0.025,
                                 m=1.5,
                                 x=0,
                                 verbose=int(verbose))

        return_dict = {'cerebellum_probability_image' : cerebellum_probability_image,
                       'parcellation_segmentation_image' : region_segmentation,
                       'parcellation_probability_images' : region_probability_images,
                       'tissue_segmentation_image' : tissue_segmentation,
                       'tissue_probability_images' : tissue_probability_images,
                       'thickness_image' : kk
                        }
        return(return_dict)

    else:
        return_dict = {'cerebellum_probability_image' : cerebellum_probability_image,
                       'parcellation_segmentation_image' : region_segmentation,
                       'parcellation_probability_images' : region_probability_images,
                       'tissue_segmentation_image' : tissue_segmentation,
                       'tissue_probability_images' : tissue_probability_images
                        }
        return(return_dict)



