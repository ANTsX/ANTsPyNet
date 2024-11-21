
import numpy as np
import ants

def brain_mra_vessel_segmentation(mra,
                                  mask=None,
                                  prediction_batch_size=16,
                                  patch_stride_length=32,
                                  antsxnet_cache_directory=None,
                                  verbose=False):

    """
    Perform MRA-TOF vessel segmentation.  Training data taken from the 
    (https://data.kitware.com/#item/58a372e48d777f0721a64dc9). 

    Arguments
    ---------
    mra : ANTsImage
        input mra image

    mask : ANTsImage
        input binary brain mask which defines the patch extraction.  If not supplied,
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
    Segmentation probability image

    Example
    -------
    >>> mra = ants.image_read("mra.nii.gz")
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import brain_extraction

    ################################
    #
    # Preprocess images
    #
    ################################

    if mask is None:
        mask = brain_extraction(mra, modality="mra",
                                antsxnet_cache_directory=antsxnet_cache_directory,
                                verbose=verbose)
        mask = ants.threshold_image(mask, 0.5, 1.1, 1, 0)

    template = ants.image_read(get_antsxnet_data("mraTemplate"))
    template_brain_mask = ants.image_read(get_antsxnet_data("mraTemplateBrainMask"))

    mra_preprocessed = ants.image_clone(mra) 
    mra_preprocessed[mask == 1] = ((mra_preprocessed[mask == 1] - mra_preprocessed[mask == 1].min()) / 
                                   (mra_preprocessed[mask == 1].max() - mra_preprocessed[mask == 1].min()))
    reg = ants.registration(template * template_brain_mask, mra_preprocessed * mask,
                            type_of_transform="antsRegistrationSyNQuick[a]",
                            verbose=verbose)
    mra_preprocessed = ants.image_clone(reg['warpedmovout'])

    patch_size = (160, 160, 160)

    if np.any(mra_preprocessed.shape < np.array(patch_size)):
        raise ValueError("Images must be > 160 voxels per dimension.")

    template_mra_prior = ants.image_read(get_antsxnet_data("mraTemplateVesselPrior"))
    template_mra_prior = ((template_mra_prior - template_mra_prior.min()) /
                          (template_mra_prior.max() - template_mra_prior.min()))
    
    ################################
    #
    # Build model and load weights
    #
    ################################

    if verbose:
        print("Load model and weights.")

    if isinstance(patch_stride_length, int):
        patch_stride_length = (patch_stride_length,) * 3

    channel_size = 2  

    model = create_unet_model_3d((*patch_size, channel_size),
                number_of_outputs=1, mode="sigmoid", 
                number_of_filters=(32, 64, 128, 256, 512),
                convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
                dropout_rate=0.0, weight_decay=0)

    weights_file_name = get_pretrained_network("mraVesselWeights_160", 
                                               antsxnet_cache_directory=antsxnet_cache_directory)
    model.load_weights(weights_file_name)

    ################################
    #
    # Extract patches
    #
    ################################

    if verbose:
        print("Extract patches.")

    mra_patches = extract_image_patches(mra_preprocessed,
                                        patch_size=patch_size,
                                        max_number_of_patches="all",
                                        stride_length=patch_stride_length,
                                        mask_image=template_brain_mask,
                                        random_seed=None,
                                        return_as_array=True)
    mra_prior_patches = extract_image_patches(template_mra_prior,
                                        patch_size=patch_size,
                                        max_number_of_patches="all",
                                        stride_length=patch_stride_length,
                                        mask_image=template_brain_mask,
                                        random_seed=None,
                                        return_as_array=True)
    total_number_of_patches = mra_patches.shape[0]

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
     
    prediction = np.zeros((total_number_of_patches, *patch_size, 1))
    for b in range(number_of_batches):
        batchX = None
        if b < number_of_batches - 1 or residual_number_of_patches == 0:
            batchX = np.zeros((prediction_batch_size, *patch_size, channel_size))
        else:
            batchX = np.zeros((residual_number_of_patches, *patch_size, channel_size))

        indices = range(b * prediction_batch_size, b * prediction_batch_size + batchX.shape[0])
        batchX[:,:,:,:,0] = mra_patches[indices,:,:,:]
        batchX[:,:,:,:,1] = mra_prior_patches[indices,:,:,:]
        
        if verbose:
            print("Predicting batch ", str(b + 1), " of ", str(number_of_batches))  
        prediction[indices,:,:,:,:] = model.predict(batchX, verbose=verbose)

    if verbose:
        print("Predict patches and reconstruct.")

    probability_image_warped = reconstruct_image_from_patches(np.squeeze(prediction[:,:,:,:,0]),
                                                       stride_length=patch_stride_length,
                                                       domain_image=template_brain_mask,
                                                       domain_image_is_mask=True)
    probability_image = ants.apply_transforms(mra, probability_image_warped,
                                              transformlist=reg['invtransforms'],
                                              whichtoinvert=[True],
                                              verbose=verbose)
    return(probability_image)

