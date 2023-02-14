import ants
import numpy as np

def whole_head_inpainting(image,
                          roi_mask,
                          modality="t1",
                          antsxnet_cache_directory=None,
                          verbose=False):

    """
    Perform in-painting for whole-head MRI

    Arguments
    ---------
    image : ANTsImage
        input MR image

    roi_mask : ANTsImage
        binary mask image

    modality : string
        Modality image type.  Options include:
            * "t1": T1-weighted MRI.
            * "flair": FLAIR MRI.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Reconstructed image.

    Example
    -------
    >>> 
    """

    from ..architectures import create_partial_convolution_unet_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import regression_match_image

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    image_size = (256, 256)
    channel_modalities = ["T1"]
    channel_size = len(channel_modalities)

    reorient_template = ants.image_read(get_antsxnet_data("oasis"))    
    template_priors = list()
    # for i in range(6):
    #     template_priors.append(ants.image_read("~/Desktop/Oasis/priors" + str(i+1) + ".nii.gz"))

    inpainting_unet = create_partial_convolution_unet_model_2d((*image_size, channel_size),
                                                                number_of_priors=0,
                                                                number_of_filters=(32, 64, 128, 256, 512, 512),
                                                                kernel_size=3)

    weights_name = ''
    if modality == "T1" or modality == "t1":
        weights_name = "wholeHeadInpaintingT1"
    elif modality == "FLAIR" or modality == "flair":
        weights_name = "wholeHeadInpaintingFLAIR"
    else:
        raise ValueError("Unavailable modality given: " + modality)

    weights_file_name = get_pretrained_network(weights_name,
        antsxnet_cache_directory=antsxnet_cache_directory)
    inpainting_unet.load_weights(weights_file_name)

    if verbose:
        print("Preprocessing:  Reorientation.")

    center_of_mass_template = np.asarray(ants.get_center_of_mass(reorient_template)).round()
    center_of_mass_image = np.asarray(ants.get_center_of_mass(image)).round()
    translation = center_of_mass_image - center_of_mass_template
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)

    image_reoriented = xfrm.apply_to_image(image, reorient_template, interpolation="linear")
    roi_mask_reoriented = xfrm.apply_to_image(roi_mask, reorient_template, interpolation="nearestneighbor")
    roi_mask_reoriented = ants.threshold_image(roi_mask_reoriented, 0, 0, 0, 1)
    roi_inverted_mask_reoriented = ants.threshold_image(roi_mask_reoriented, 0, 0, 1, 0)

    geoms = ants.label_geometry_measures(roi_mask_reoriented)
    if geoms.shape[0] != 1:
        raise ValueError("ROI is not specified correctly.")
    lower_slice = int(geoms['BoundingBoxLower_y'])
    upper_slice = int(geoms['BoundingBoxUpper_y'])
    number_of_slices = upper_slice - lower_slice + 1
       
    if verbose:
        print("Preprocessing:  Slicing data.")

    batchX = np.zeros((number_of_slices, *image_size, channel_size))
    batchXMask = np.zeros((number_of_slices, *image_size, channel_size))
    if len(template_priors) > 0:
        batchXPriors = np.zeros((number_of_slices, *image_size, len(template_priors)))

    for i in range(number_of_slices):
        index = lower_slice + i

        mask_slice = ants.slice_image(roi_inverted_mask_reoriented, axis=1, idx=index, collapse_strategy=1)
        mask_slice = pad_or_crop_image_to_size(mask_slice, image_size)
        mask_slice_array = mask_slice.numpy()

        slice = ants.slice_image(image_reoriented, axis=1, idx=index, collapse_strategy=1)
        slice = pad_or_crop_image_to_size(slice, image_size)
        slice = mask_slice * (slice - slice.min()) / (slice.max() - slice.min())

        slice[mask_slice == 0] = 1
        slice_array = slice.numpy()

        for j in range(channel_size):
            batchX[i,:,:,j] = slice_array
            batchXMask[i,:,:,j] = mask_slice_array

        for j in range(len(template_priors)):
            prior_slice = ants.slice_image(template_priors[j], axis=1, idx=index, collapse_strategy=1)
            prior_slice = pad_or_crop_image_to_size(prior_slice, image_size)
            batchXPriors[i,:,:,j] = prior_slice.numpy()
        
    if verbose:
        print("Prediction.")

    predicted_data = None
    predicted_data = inpainting_unet.predict([batchX, batchXMask], verbose=int(verbose))
    # predicted_data = inpainting_unet.predict([batchX, batchXMask, batchXPriors], verbose=int(verbose))
    predicted_data[batchXMask == 1] = batchX[batchXMask == 1]

    if verbose:
        print("Post-processing:  Slicing data.")

    image_reoriented_array = image_reoriented.numpy()
    for i in range(number_of_slices):
        index = lower_slice + i

        slice = ants.slice_image(image_reoriented, axis=1, idx=index, collapse_strategy=1)
        mask_slice = ants.slice_image(roi_inverted_mask_reoriented, axis=1, idx=index, collapse_strategy=1)
        predicted_slice = ants.from_numpy(np.squeeze(predicted_data[i,:,:,0]), origin=slice.origin, 
            spacing=slice.spacing, direction=slice.direction)
        predicted_slice = pad_or_crop_image_to_size(predicted_slice, slice.shape)
        ants.set_origin(predicted_slice, slice.origin)
        predicted_slice = regression_match_image(predicted_slice, slice)  
        image_reoriented_array[:,index,:] = predicted_slice.numpy()
        
    inpainted_image = ants.from_numpy(np.squeeze(image_reoriented_array), 
       origin=image_reoriented.origin, spacing=image_reoriented.spacing,
       direction=image_reoriented.direction)
     
    if verbose == True:
        print("Post-processing:  reorienting to original space.")

    xfrm_inv = xfrm.invert()
    inpainted_image = xfrm_inv.apply_to_image(inpainted_image, image, interpolation="linear")  
    inpainted_image = ants.copy_image_info(image, inpainted_image)
    inpainted_image[roi_mask == 0] = image[roi_mask == 0]

    return(inpainted_image)


