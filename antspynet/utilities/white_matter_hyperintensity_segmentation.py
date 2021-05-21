
import numpy as np
import ants

def sysu_media_wmh_segmentation(flair,
                                t1=None,
                                use_ensemble=True,
                                antsxnet_cache_directory=None,
                                verbose=False):

    """
    Perform WMH segmentation using the winning submission in the MICCAI
    2017 challenge by the sysu_media team using FLAIR or T1/FLAIR.  The
    MICCAI challenge is discussed in

    https://pubmed.ncbi.nlm.nih.gov/30908194/

    with the sysu_media's team entry is discussed in

     https://pubmed.ncbi.nlm.nih.gov/30125711/

    with the original implementation available here:

    https://github.com/hongweilibran/wmh_ibbmTum

    Arguments
    ---------
    flair : ANTsImage
        input 3-D FLAIR brain image (not skull-stripped).

    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    use_ensemble : boolean
        check whether to use all 3 sets of weights.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> image = ants.image_read("flair.nii.gz")
    >>> probability_mask = sysu_media_wmh_segmentation(image)
    """

    from ..architectures import create_sysu_media_unet_model_2d
    from ..utilities import brain_extraction
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities import pad_or_crop_image_to_size

    if flair.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    image_size = (200, 200)

    ################################
    #
    # Preprocess images
    #
    ################################

    flair_preprocessed = flair
    number_of_channels = 1

    if t1 is not None:
        t1_preprocessed = t1
        number_of_channels = 2

    reference_image = ants.make_image((170, 256, 256),
                                       voxval=1,
                                       spacing=(1, 1, 1),
                                       origin=(0, 0, 0),
                                       direction=np.identity(3))
    center_of_mass_reference = ants.get_center_of_mass(reference_image)
    center_of_mass_image = ants.get_center_of_mass(flair_preprocessed)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    flair_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, flair_preprocessed, reference_image)

    if t1 is not None:
        t1_preprocessed_warped = ants.apply_ants_transform_to_image(xfrm, t1_preprocessed, reference_image)

    ################################
    #
    # Gaussian normalize intensity based on brain mask
    #
    ################################

    mean_flair = flair_preprocessed_warped.mean()
    std_flair = flair_preprocessed_warped.std()
    flair_preprocessed_warped = (flair_preprocessed_warped - mean_flair) / std_flair
    if number_of_channels == 2:
        mean_t1 = t1_preprocessed_warped.mean()
        std_t1 = t1_preprocessed_warped.std()
        t1_preprocessed_warped = (t1_preprocessed_warped - mean_t1) / std_t1

    ################################
    #
    # Build models and load weights
    #
    ################################

    number_of_models = 1
    if use_ensemble == True:
        number_of_models = 3

    if verbose == True:
        print("White matter hyperintensity:  retrieving model weights.")

    unet_models = list()
    for i in range(number_of_models):
        if number_of_channels == 1:
            weights_file_name = get_pretrained_network("sysuMediaWmhFlairOnlyModel" + str(i), antsxnet_cache_directory=antsxnet_cache_directory)
        else:
            weights_file_name = get_pretrained_network("sysuMediaWmhFlairT1Model" + str(i), antsxnet_cache_directory=antsxnet_cache_directory)
        unet_models.append(create_sysu_media_unet_model_2d((200, 200, number_of_channels)))
        unet_models[i].load_weights(weights_file_name)

    ################################
    #
    # Extract slices
    #
    ################################

    dimensions_to_predict = [2]

    total_number_of_slices = 0
    for d in range(len(dimensions_to_predict)):
        total_number_of_slices += flair_preprocessed_warped.shape[dimensions_to_predict[d]]

    batchX = np.zeros((total_number_of_slices, 200, 200, number_of_channels))

    slice_count = 0
    for d in range(len(dimensions_to_predict)):
        number_of_slices = flair_preprocessed_warped.shape[dimensions_to_predict[d]]

        if verbose == True:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            flair_slice = pad_or_crop_image_to_size(ants.slice_image(
                flair_preprocessed_warped, dimensions_to_predict[d], i), image_size)
            batchX[slice_count,:,:,0] = flair_slice.numpy()
            if number_of_channels == 2:
                t1_slice = pad_or_crop_image_to_size(ants.slice_image(
                    t1_preprocessed_warped, dimensions_to_predict[d], i), image_size)
                batchX[slice_count,:,:,1] = t1_slice.numpy()
            slice_count += 1

    ants.image_write(ants.from_numpy(np.squeeze(batchX[:,:,:,1])), "~/Desktop/t1Py.nii.gz") 
    ants.image_write(ants.from_numpy(np.squeeze(batchX[:,:,:,0])), "~/Desktop/flairPy.nii.gz") 

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    if verbose == True:
        print("Prediction.")

    prediction = unet_models[0].predict(np.rot90(batchX, axes=(1, 2), k=1), verbose=verbose)
    if number_of_models > 1:
        for i in range(1, number_of_models, 1):
            prediction += unet_models[i].predict(np.rot90(batchX, axes=(1, 2), k=1), verbose=verbose)
    prediction /= number_of_models
    prediction = np.rot90(prediction, axes=(1, 2), k=-1)

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    prediction_image_average = ants.image_clone(flair_preprocessed_warped) * 0

    current_start_slice = 0
    for d in range(len(dimensions_to_predict)):
        current_end_slice = current_start_slice + flair_preprocessed_warped.shape[dimensions_to_predict[d]]
        which_batch_slices = range(current_start_slice, current_end_slice)
        prediction_per_dimension = prediction[which_batch_slices,:,:,:]
        prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
        prediction_image = ants.copy_image_info(flair_preprocessed_warped,
          pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
            flair_preprocessed_warped.shape))
        prediction_image_average = prediction_image_average + (prediction_image - prediction_image_average) / (d + 1)
        current_start_slice = current_end_slice

    probability_image = ants.apply_ants_transform_to_image(
        ants.invert_ants_transform(xfrm), prediction_image_average, flair)

    return(probability_image)

def ew_david(flair,
             t1,
             do_preprocessing=True,
             which_model="sysu",
             which_axes="max",
             number_of_simulations=0,
             sd_affine=0.01,
             antsxnet_cache_directory=None,
             verbose=False):

    """
    Perform White matter hypterintensity probabilistic segmentation
    using deep learning

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * brain extraction, and
       * affine registration to MNI.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    \code{doPreprocessing = TRUE}

    Arguments
    ---------
    flair : ANTsImage
        input 3-D FLAIR brain image (not skull-stripped).

    t1 : ANTsImage
        input 3-D T1 brain image (not skull-stripped).

    do_preprocessing : boolean
        perform n4 bias correction?

    which_model : string
       one of "sysu", "sysuPlus", "sysuPlusSeg", "sysuWithSite".

    which_axes : string or scalar or tuple/vector
        apply 2-D model to 1 or more axes.  In addition to a scalar
        or vector, e.g., which_axes = (0, 2), one can use "max" for the
        axis with maximum anisotropy (default) or "all" for all axes.

    number_of_simulations : integer
        Number of random affine perturbations to transform the input.

    sd_affine : float
        Define the standard deviation of the affine transformation parameter.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> image = ants.image_read("flair.nii.gz")
    >>> probability_mask = sysu_media_wmh_segmentation(image)
    """

    from ..architectures import create_unet_model_2d
    from ..architectures import create_unet_model_3d
    from ..utilities import deep_atropos
    from ..utilities import get_pretrained_network
    from ..utilities import preprocess_brain_image
    from ..utilities import brain_extraction
    from ..utilities import extract_image_patches
    from ..utilities import randomly_transform_image_data
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import pad_or_crop_image_to_size

    do_t1_only = False
    do_flair_only = False

    if flair is None and t1 is not None:
        do_t1_only = True
    elif flair is not None and t1 is None:
        do_flair_only = True

    use_t1_segmentation = False
    if "Seg" in which_model:
        if do_flair_only:
            raise ValueError("Segmentation requires T1.")
        else:
            use_t1_segmentation = True

    if use_t1_segmentation and do_preprocessing == False:
        raise ValueError("Using the t1 segmentation requires do_preprocessing=True.")

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    do_slicewise = True

    if do_slicewise == False:

        raise ValueError("Not available.")

        # ################################
        # #
        # # Preprocess images
        # #
        # ################################

        # t1_preprocessed = t1
        # t1_preprocessing = None
        # if do_preprocessing == True:
        #     t1_preprocessing = preprocess_brain_image(t1,
        #         truncate_intensity=(0.01, 0.99),
        #         brain_extraction_modality="t1",
        #         template="croppedMni152",
        #         template_transform_type="antsRegistrationSyNQuickRepro[a]",
        #         do_bias_correction=True,
        #         do_denoising=False,
        #         antsxnet_cache_directory=antsxnet_cache_directory,
        #         verbose=verbose)
        #     t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

        # flair_preprocessed = flair
        # if do_preprocessing == True:
        #     flair_preprocessing = preprocess_brain_image(flair,
        #         truncate_intensity=(0.01, 0.99),
        #         brain_extraction_modality="t1",
        #         do_bias_correction=True,
        #         do_denoising=False,
        #         antsxnet_cache_directory=antsxnet_cache_directory,
        #         verbose=verbose)
        #     flair_preprocessed = ants.apply_transforms(fixed=t1_preprocessed,
        #         moving=flair_preprocessing["preprocessed_image"],
        #         transformlist=t1_preprocessing['template_transforms']['fwdtransforms'])
        #     flair_preprocessed = flair_preprocessed * t1_preprocessing['brain_mask']

        # ################################
        # #
        # # Build model and load weights
        # #
        # ################################

        # patch_size = (112, 112, 112)
        # stride_length = (t1_preprocessed.shape[0] - patch_size[0],
        #                 t1_preprocessed.shape[1] - patch_size[1],
        #                 t1_preprocessed.shape[2] - patch_size[2])

        # classes = ("background", "wmh" )
        # number_of_classification_labels = len(classes)
        # labels = (0, 1)

        # image_modalities = ("T1", "FLAIR")
        # channel_size = len(image_modalities)

        # unet_model = create_unet_model_3d((*patch_size, channel_size),
        #     number_of_outputs = number_of_classification_labels,
        #     number_of_layers = 4, number_of_filters_at_base_layer = 16, dropout_rate = 0.0,
        #     convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
        #     weight_decay = 1e-5, additional_options=("attentionGating"))

        # weights_file_name = get_pretrained_network("ewDavidWmhSegmentationWeights",
        #     antsxnet_cache_directory=antsxnet_cache_directory)
        # unet_model.load_weights(weights_file_name)

        # ################################
        # #
        # # Do prediction and normalize to native space
        # #
        # ################################

        # if verbose == True:
        #     print("ew_david:  prediction.")

        # batchX = np.zeros((8, *patch_size, channel_size))

        # t1_preprocessed = (t1_preprocessed - t1_preprocessed.mean()) / t1_preprocessed.std()
        # t1_patches = extract_image_patches(t1_preprocessed, patch_size=patch_size,
        #                                     max_number_of_patches="all", stride_length=stride_length,
        #                                     return_as_array=True)
        # batchX[:,:,:,:,0] = t1_patches

        # flair_preprocessed = (flair_preprocessed - flair_preprocessed.mean()) / flair_preprocessed.std()
        # flair_patches = extract_image_patches(flair_preprocessed, patch_size=patch_size,
        #                                     max_number_of_patches="all", stride_length=stride_length,
        #                                     return_as_array=True)
        # batchX[:,:,:,:,1] = flair_patches

        # predicted_data = unet_model.predict(batchX, verbose=verbose)

        # probability_images = list()
        # for i in range(len(labels)):
        #     print("Reconstructing image", classes[i])
        #     reconstructed_image = reconstruct_image_from_patches(predicted_data[:,:,:,:,i],
        #         domain_image=t1_preprocessed, stride_length=stride_length)

        #     if do_preprocessing == True:
        #         probability_images.append(ants.apply_transforms(fixed=t1,
        #             moving=reconstructed_image,
        #             transformlist=t1_preprocessing['template_transforms']['invtransforms'],
        #             whichtoinvert=[True], interpolator="linear", verbose=verbose))
        #     else:
        #         probability_images.append(reconstructed_image)

        # return(probability_images[1])

    else:  # do_slicewise

        ################################
        #
        # Preprocess images
        #
        ################################

        t1_preprocessed = None
        t1_preprocessing = None
        brain_mask = None
        if t1 is not None:
            if do_preprocessing == True:
                t1_preprocessing = preprocess_brain_image(t1,
                    truncate_intensity=(0.01, 0.99),
                    brain_extraction_modality="t1",
                    do_bias_correction=True,
                    do_denoising=False,
                    antsxnet_cache_directory=antsxnet_cache_directory,
                    verbose=verbose)
                brain_mask = ants.threshold_image(t1_preprocessing["brain_mask"], 0.5, 1, 1, 0)
                t1_preprocessed = t1_preprocessing["preprocessed_image"] * brain_mask

        t1_segmentation = None
        if use_t1_segmentation:
            atropos_seg = deep_atropos(t1, do_preprocessing=True, verbose=verbose)
            t1_segmentation = atropos_seg['segmentation_image']

        flair_preprocessed = None
        if flair is not None:
            flair_preprocessed = flair
            if do_preprocessing == True:
                if brain_mask is None:  
                    flair_preprocessing = preprocess_brain_image(flair,
                        truncate_intensity=(0.01, 0.99),
                        brain_extraction_modality="flair",
                        do_bias_correction=True,
                        do_denoising=False,
                        antsxnet_cache_directory=antsxnet_cache_directory,
                        verbose=verbose)
                    brain_mask = flair_preprocessing["brain_mask"]
                else:
                    flair_preprocessing = preprocess_brain_image(flair,
                        truncate_intensity=(0.01, 0.99),
                        brain_extraction_modality=None,
                        do_bias_correction=True,
                        do_denoising=False,
                        antsxnet_cache_directory=antsxnet_cache_directory,
                        verbose=verbose)
                flair_preprocessed = flair_preprocessing["preprocessed_image"] * brain_mask

        if t1_preprocessed is not None:
            resampling_params = list(ants.get_spacing(t1_preprocessed))
        else:
            resampling_params = list(ants.get_spacing(flair_preprocessed))

        do_resampling = False
        for d in range(len(resampling_params)):
            if resampling_params[d] < 0.8:
                resampling_params[d] = 1.0
                do_resampling = True

        resampling_params = tuple(resampling_params)

        if do_resampling:
            if flair_preprocessed is not None:
                flair_preprocessed = ants.resample_image(flair_preprocessed, resampling_params, use_voxels=False, interp_type=0)
            if t1_preprocessed is not None:
                t1_preprocessed = ants.resample_image(t1_preprocessed, resampling_params, use_voxels=False, interp_type=0)
            if t1_segmentation is not None:
                t1_segmentation = ants.resample_image(t1_segmentation, resampling_params, use_voxels=False, interp_type=1)
            if brain_mask is not None:
                brain_mask = ants.resample_image(brain_mask, resampling_params, use_voxels=False, interp_type=1)

        ################################
        #
        # Build model and load weights
        #
        ################################

        template_size = (208, 208)

        image_modalities = ("T1", "FLAIR")
        if do_flair_only:
            image_modalities=("FLAIR",)
        elif do_t1_only:
            image_modalities=("T1",)
        if use_t1_segmentation:
            image_modalities = (*image_modalities, "T1Seg")
        channel_size = len(image_modalities)

        if which_model == "sysu":
            unet_model = create_unet_model_2d((*template_size, channel_size),
                number_of_outputs=1, mode="sigmoid",
                number_of_filters=(64, 96, 128, 256, 512), dropout_rate=0.0,
                convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
                weight_decay=0, additional_options=("initialConvolutionKernelSize[5]",))
        elif which_model == "sysuWithSite":
            unet_model = create_unet_model_2d((*template_size, channel_size),
                number_of_outputs=1, mode="sigmoid",
                scalar_output_size=3, scalar_output_activation="softmax",
                number_of_filters=(64, 96, 128, 256, 512), dropout_rate=0.0,
                convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
                weight_decay=0, additional_options=("initialConvolutionKernelSize[5]",))
        else:
            unet_model = create_unet_model_2d((*template_size, channel_size),
                number_of_outputs=1, mode="sigmoid",
                number_of_filters=(64, 96, 128, 256, 512), dropout_rate=0.0,
                convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
                weight_decay=1e-5,
                additional_options=("nnUnetActivationStyle", "attentionGating", "initialConvolutionKernelSize[5]",))

        if verbose == True:
            print("ewDavid:  retrieving model weights.")

        weights_file_name = None
        if which_model == "sysu" and flair is not None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysu", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysu" and flair is None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuT1Only", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysu" and flair is not None and t1 is None:
            weights_file_name = get_pretrained_network("ewDavidSysuFlairOnly", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuPlus" and flair is not None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuPlus", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuPlus" and flair is None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuPlusT1Only", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuPlus" and flair is not None and t1 is None:
            weights_file_name = get_pretrained_network("ewDavidSysuPlusFlairOnly", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuPlusSeg" and flair is not None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuPlusSeg", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuPlusSeg" and flair is None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuPlusSegT1Only", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuWithSite" and flair is not None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuWithSite", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuWithSite" and flair is None and t1 is not None:
            weights_file_name = get_pretrained_network("ewDavidSysuWithSiteT1Only", antsxnet_cache_directory=antsxnet_cache_directory)
        elif which_model == "sysuWithSite" and flair is not None and t1 is None:
            weights_file_name = get_pretrained_network("ewDavidSysuWithSiteFlairOnly", antsxnet_cache_directory=antsxnet_cache_directory)
        else:
            raise ValueError("Incorrect model specification or image combination.")

        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Data augmentation and extract slices
        #
        ################################

        wmh_probability_image = None
        if t1 is not None:
            wmh_probability_image = ants.image_clone(t1_preprocessed) * 0
        else:
            wmh_probability_image = ants.image_clone(flair_preprocessed) * 0

        wmh_site = np.array([0, 0, 0])

        data_augmentation = None
        if number_of_simulations > 0:
            if do_flair_only:
                data_augmentation = randomly_transform_image_data(
                    reference_image=flair_preprocessed,
                    input_image_list=[[flair_preprocessed]],
                    number_of_simulations=number_of_simulations,
                    transform_type='affine',
                    sd_affine=sd_affine,
                    input_image_interpolator='linear')
            elif do_t1_only:
                if use_t1_segmentation:
                    data_augmentation = randomly_transform_image_data(
                        reference_image=t1_preprocessed,
                        input_image_list=[[t1_preprocessed]],
                        segmentation_image_list=[t1_segmentation],
                        number_of_simulations=number_of_simulations,
                        transform_type='affine',
                        sd_affine=sd_affine,
                        input_image_interpolator='linear',
                        segmentation_image_interpolator='nearestNeighbor')
                else:
                    data_augmentation = randomly_transform_image_data(
                        reference_image=t1_preprocessed,
                        input_image_list=[[t1_preprocessed]],
                        number_of_simulations=number_of_simulations,
                        transform_type='affine',
                        sd_affine=sd_affine,
                        input_image_interpolator='linear')
            else:
                if use_t1_segmentation:
                    data_augmentation = randomly_transform_image_data(
                        reference_image=t1_preprocessed,
                        input_image_list=[[flair_preprocessed, t1_preprocessed]],
                        segmentation_image_list=[t1_segmentation],
                        number_of_simulations=number_of_simulations,
                        transform_type='affine',
                        sd_affine=sd_affine,
                        input_image_interpolator='linear',
                        segmentation_image_interpolator='nearestNeighbor')
                else:
                    data_augmentation = randomly_transform_image_data(
                        reference_image=t1_preprocessed,
                        input_image_list=[[flair_preprocessed, t1_preprocessed]],
                        number_of_simulations=number_of_simulations,
                        transform_type='affine',
                        sd_affine=sd_affine,
                        input_image_interpolator='linear')

        dimensions_to_predict = list((0,))
        if which_axes == "max":
            spacing = ants.get_spacing(wmh_probability_image)
            dimensions_to_predict = (spacing.index(max(spacing)),)
        elif which_axes == "all":
            dimensions_to_predict = list(range(3))
        else:
            if isinstance(which_axes, int):
                dimensions_to_predict = list((which_axes,))
            else:
                dimensions_to_predict = list(which_axes)

        total_number_of_slices = 0
        for d in range(len(dimensions_to_predict)):
            total_number_of_slices += wmh_probability_image.shape[dimensions_to_predict[d]]

        batchX = np.zeros((total_number_of_slices, *template_size, channel_size))

        for n in range(number_of_simulations + 1):

            batch_flair = flair_preprocessed
            batch_t1 = t1_preprocessed
            batch_t1_segmentation = t1_segmentation

            if n > 0:
                if do_flair_only:
                    batch_flair = data_augmentation['simulated_images'][n-1][0]
                elif do_t1_only:
                    batch_t1 = data_augmentation['simulated_images'][n-1][0]
                else:
                    batch_flair = data_augmentation['simulated_images'][n-1][0]
                    batch_t1 = data_augmentation['simulated_images'][n-1][1]
                if use_t1_segmentation:
                    batch_t1_segmentation = data_augmentation['simulated_segmentation_images'][n-1]

            if batch_flair is not None:
                batch_flair = (batch_flair - batch_flair[brain_mask != 0].mean()) / batch_flair[brain_mask != 0].std()
            if batch_t1 is not None:
                batch_t1 = (batch_t1 - batch_t1[brain_mask != 0].mean()) / batch_t1[brain_mask != 0].std()

            slice_count = 0
            for d in range(len(dimensions_to_predict)):

                number_of_slices = None
                if batch_t1 is not None:
                    number_of_slices = batch_t1.shape[dimensions_to_predict[d]]
                else:
                    number_of_slices = batch_flair.shape[dimensions_to_predict[d]]

                if verbose == True:
                    print("Extracting slices for dimension ", dimensions_to_predict[d])

                for i in range(number_of_slices):

                    channel_count = 0
                    if batch_flair is not None:
                        flair_slice = pad_or_crop_image_to_size(ants.slice_image(batch_flair, dimensions_to_predict[d], i), template_size)
                        batchX[slice_count,:,:,channel_count] = flair_slice.numpy()
                        channel_count += 1
                    if batch_t1 is not None:
                        t1_slice = pad_or_crop_image_to_size(ants.slice_image(batch_t1, dimensions_to_predict[d], i), template_size)
                        batchX[slice_count,:,:,channel_count] = t1_slice.numpy()
                        channel_count += 1
                    if t1_segmentation is not None:
                        t1_segmentation_slice = pad_or_crop_image_to_size(ants.slice_image(batch_t1_segmentation, dimensions_to_predict[d], i), template_size)
                        batchX[slice_count,:,:,channel_count] = t1_segmentation_slice.numpy() / 6

                    slice_count += 1

            ################################
            #
            # Do prediction and then restack into the image
            #
            ################################

            if verbose == True:
                if n == 0:
                    print("Prediction")
                else:
                    print("Prediction (simulation " + str(n) + ")")

            prediction = unet_model.predict(batchX, verbose=verbose)

            permutations = list()
            permutations.append((0, 1, 2))
            permutations.append((1, 0, 2))
            permutations.append((1, 2, 0))

            prediction_image_average = ants.image_clone(wmh_probability_image) * 0

            current_start_slice = 0
            for d in range(len(dimensions_to_predict)):
                current_end_slice = current_start_slice + wmh_probability_image.shape[dimensions_to_predict[d]]
                which_batch_slices = range(current_start_slice, current_end_slice)
                if isinstance(prediction, list):
                    prediction_per_dimension = prediction[0][which_batch_slices,:,:,0]
                else:
                    prediction_per_dimension = prediction[which_batch_slices,:,:,0]
                prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
                prediction_image = ants.copy_image_info(wmh_probability_image,
                    pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
                    wmh_probability_image.shape))
                prediction_image_average = prediction_image_average + (prediction_image - prediction_image_average) / (d + 1)
                current_start_slice = current_end_slice

            wmh_probability_image = wmh_probability_image + (prediction_image_average - wmh_probability_image) / (n + 1)
            if isinstance(prediction, list):
                wmh_site = wmh_site + (np.mean(prediction[1], axis=0) - wmh_site) / (n + 1)

        if do_resampling:
            if t1 is not None:
                wmh_probability_image = ants.resample_image_to_target(wmh_probability_image, t1)
            if flair is not None:
                wmh_probability_image = ants.resample_image_to_target(wmh_probability_image, flair)

        if isinstance(prediction, list):
            return([wmh_probability_image, wmh_site])
        else:
            return(wmh_probability_image)
