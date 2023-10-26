
import numpy as np
import ants

def el_bicho(ventilation_image,
             mask,
             use_coarse_slices_only=True,
             antsxnet_cache_directory=None,
             verbose=False):

    """
    Perform functional lung segmentation using hyperpolarized gases.

    https://pubmed.ncbi.nlm.nih.gov/30195415/

    Arguments
    ---------
    ventilation_image : ANTsImage
        input ventilation image.

    mask : ANTsImage
        input mask.

    use_coarse_slices_only : boolean
        If True, apply network only in the dimension of greatest slice thickness.
        If False, apply to all dimensions and average the results.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Ventilation segmentation and corresponding probability images

    Example
    -------
    >>> image = ants.image_read("ventilation.nii.gz")
    >>> mask = ants.image_read("mask.nii.gz")
    >>> lung_seg = el_bicho(image, mask, use_coarse_slices=True, verbose=False)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import pad_or_crop_image_to_size

    if ventilation_image.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    if ventilation_image.shape != mask.shape:
        raise ValueError("Ventilation image and mask size are not the same size.")

    ################################
    #
    # Preprocess image
    #
    ################################

    template_size = (256, 256)
    classes = (0, 1, 2, 3, 4)
    number_of_classification_labels = len(classes)

    image_modalities = ("Ventilation", "Mask")
    channel_size = len(image_modalities)

    preprocessed_image = (ventilation_image - ventilation_image.mean()) / ventilation_image.std()
    ants.set_direction(preprocessed_image, np.identity(3))

    mask_identity = ants.image_clone(mask)
    ants.set_direction(mask_identity, np.identity(3))

    ################################
    #
    # Build models and load weights
    #
    ################################

    unet_model = create_unet_model_2d((*template_size, channel_size),
        number_of_outputs=number_of_classification_labels,
        number_of_layers=4, number_of_filters_at_base_layer=32, dropout_rate=0.0,
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        weight_decay=1e-5, additional_options=("attentionGating"))

    if verbose == True:
        print("El Bicho: retrieving model weights.")

    weights_file_name = get_pretrained_network("elBicho", antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Extract slices
    #
    ################################

    spacing = ants.get_spacing(preprocessed_image)
    dimensions_to_predict = (spacing.index(max(spacing)),)
    if use_coarse_slices_only == False:
        dimensions_to_predict = list(range(3))

    total_number_of_slices = 0
    for d in range(len(dimensions_to_predict)):
        total_number_of_slices += preprocessed_image.shape[dimensions_to_predict[d]]

    batchX = np.zeros((total_number_of_slices, *template_size, channel_size))

    slice_count = 0
    for d in range(len(dimensions_to_predict)):
        number_of_slices = preprocessed_image.shape[dimensions_to_predict[d]]

        if verbose == True:
            print("Extracting slices for dimension ", dimensions_to_predict[d], ".")

        for i in range(number_of_slices):
            ventilation_slice = pad_or_crop_image_to_size(ants.slice_image(preprocessed_image, dimensions_to_predict[d], i), template_size)
            batchX[slice_count,:,:,0] = ventilation_slice.numpy()

            mask_slice = pad_or_crop_image_to_size(ants.slice_image(mask_identity, dimensions_to_predict[d], i), template_size)
            batchX[slice_count,:,:,1] = mask_slice.numpy()

            slice_count += 1

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    if verbose == True:
        print("Prediction.")

    prediction = unet_model.predict(batchX, verbose=verbose)

    permutations = list()
    permutations.append((0, 1, 2))
    permutations.append((1, 0, 2))
    permutations.append((1, 2, 0))

    probability_images = list()
    for l in range(number_of_classification_labels):
        probability_images.append(ants.image_clone(mask) * 0)

    current_start_slice = 0
    for d in range(len(dimensions_to_predict)):
        current_end_slice = current_start_slice + preprocessed_image.shape[dimensions_to_predict[d]]
        which_batch_slices = range(current_start_slice, current_end_slice)

        for l in range(number_of_classification_labels):
            prediction_per_dimension = prediction[which_batch_slices,:,:,l]
            prediction_array = np.transpose(np.squeeze(prediction_per_dimension), permutations[dimensions_to_predict[d]])
            prediction_image = ants.copy_image_info(ventilation_image,
                pad_or_crop_image_to_size(ants.from_numpy(prediction_array),
                ventilation_image.shape))
            probability_images[l] = probability_images[l] + (prediction_image - probability_images[l]) / (d + 1)

        current_start_slice = current_end_slice + 1

    ################################
    #
    # Convert probability images to segmentation
    #
    ################################

    image_matrix = ants.image_list_to_matrix(probability_images[1:(len(probability_images))], mask * 0 + 1)
    background_foreground_matrix = np.stack([ants.image_list_to_matrix([probability_images[0]], mask * 0 + 1),
                                            np.expand_dims(np.sum(image_matrix, axis=0), axis=0)])
    foreground_matrix = np.argmax(background_foreground_matrix, axis=0)
    segmentation_matrix = (np.argmax(image_matrix, axis=0) + 1) * foreground_matrix
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), mask * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return(return_dict)

def lung_pulmonary_artery_segmentation(ct,
                                       lung_mask=None,
                                       prediction_batch_size=16,
                                       patch_stride_length=32,
                                       antsxnet_cache_directory=None,
                                       verbose=False):

    """
    Perform pulmonary artery segmentation.  Training data taken from the 
    PARSE2022 challenge (Luo, Gongning, et al. "Efficient automatic segmentation 
    for multi-level pulmonary arteries: The PARSE challenge." 
    https://arxiv.org/abs/2304.03708).

    Arguments
    ---------
    ct : ANTsImage
        input ct image

    lung_mask : ANTsImage
        input binary lung mask which defines the patch extraction.  If not supplied,
        one is estimated.

    prediction_batch_size : int
        Control memory usage for prediction.  More consequential for GPU-usage.

    patch_stride_length : 3-D tuple or int
        Dictates the stride length for accumulating predicting patches.    

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    WMH segmentation probability image

    Example
    -------
    >>> ct = ants.image_read("ct.nii.gz")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import get_pretrained_network
    from ..utilities import lung_extraction

    patch_size = (160, 160, 160)

    if np.any(ct.shape < np.array(patch_size)):
        raise ValueError("Images must be > 160 voxels per dimension.")

    ################################
    #
    # Preprocess images
    #
    ################################

    if lung_mask is None:
        lung_mask = ants.threshold_image(lung_extraction(ct, modality="ct", verbose=verbose), 1, 3, 1, 0)
    ct_preprocessed = ants.image_clone(ct)
    ct_preprocessed = (ct_preprocessed + 800) / (500 + 800)
    ct_preprocessed[ct_preprocessed > 1.0] = 1.0
    ct_preprocessed[ct_preprocessed < 0.0] = 0.0

    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Load model and weights.")

    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3

    number_of_classification_labels = 2
    channel_size = 1  

    model = create_unet_model_3d((*patch_size, channel_size),
                number_of_outputs=number_of_classification_labels, mode="sigmoid", 
                number_of_filters=(32, 64, 128, 256, 512),
                convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
                dropout_rate=0.0, weight_decay=0)

    weights_file_name = get_pretrained_network("pulmonaryArteryWeights", antsxnet_cache_directory=antsxnet_cache_directory)
    model.load_weights(weights_file_name)

    ################################
    #
    # Extract patches
    #
    ################################

    if verbose:
        print("Extract patches.")

    ct_patches = extract_image_patches(ct_preprocessed,
                                       patch_size=patch_size,
                                       max_number_of_patches="all",
                                       stride_length=patch_stride_length,
                                       mask_image=lung_mask,
                                       random_seed=None,
                                       return_as_array=True)
    total_number_of_patches = ct_patches.shape[0]

    ################################
    #
    # Do prediction and then restack into the image
    #
    ################################

    number_of_batches = total_number_of_patches // prediction_batch_size
    residual_number_of_patches = total_number_of_patches - number_of_batches * prediction_batch_size
    if residual_number_of_patches > 0:
        number_of_batches = number_of_batches + 1

    if verbose:
        print("  Total number of patches: ", str(total_number_of_patches))
        print("  Prediction batch size: ", str(prediction_batch_size))
        print("  Number of batches: ", str(number_of_batches))
     
    prediction = np.zeros((total_number_of_patches, *patch_size, 2))
    for b in range(number_of_batches):
        batchX = None
        if b < number_of_batches - 1 or residual_number_of_patches == 0:
            batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
        else:
            batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

        indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
        batchX[:,:,:,:,0] = ct_patches[indices,:,:,:]
        
        if verbose:
            print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))  
        prediction[indices,:,:,:,:] = model.predict(batchX, verbose=verbose)

    if verbose:
        print("Predict patches and reconstruct.")

    probability_image = reconstruct_image_from_patches(np.squeeze(prediction[:,:,:,:,1]),
                                                       stride_length=patch_stride_length,
                                                       domain_image=lung_mask,
                                                       domain_image_is_mask=True)
    return(probability_image)
