import ants
import numpy as np

def whole_head_inpainting(image,
                          roi_mask,
                          modality="t1",
                          slicewise=True,
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

    slicewise : boolean
        Two models per modality are available for processing the data.  One model
        is based on training/prediction using 2-D axial slice data whereas the
        other uses 64x64x64 patches.

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
    from ..architectures import create_partial_convolution_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import regression_match_image
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if slicewise:

        image_size = (256, 256)
        channel_size = 1

        if verbose:
            print("Preprocessing:  Reorientation.")

        reorient_template = ants.image_read(get_antsxnet_data("oasis"))

        center_of_mass_template = np.asarray(ants.get_center_of_mass(reorient_template))
        center_of_mass_image = np.asarray(ants.get_center_of_mass(image))
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
            print("Preprocessing:  Slicing data.")

        batchX = np.zeros((number_of_slices, *image_size, channel_size))
        batchXMask = np.zeros((number_of_slices, *image_size, channel_size))

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

        if verbose:
            print("Prediction.")

        predicted_data = inpainting_unet.predict([batchX, batchXMask], verbose=int(verbose))
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
            predicted_slice = regression_match_image(predicted_slice, slice, mask=mask_slice)
            image_reoriented_array[:,index,:] = predicted_slice.numpy()

        inpainted_image = ants.from_numpy(np.squeeze(image_reoriented_array),
           origin=image_reoriented.origin, spacing=image_reoriented.spacing,
           direction=image_reoriented.direction)

        if verbose:
            print("Post-processing:  reorienting to original space.")

        xfrm_inv = xfrm.invert()
        inpainted_image = xfrm_inv.apply_to_image(inpainted_image, image, interpolation="linear")
        inpainted_image = ants.copy_image_info(image, inpainted_image)
        inpainted_image[roi_mask == 0] = image[roi_mask == 0]

        return(inpainted_image)

    else:

        image_size = (256, 256, 256)
        patch_size = (64, 64, 64)
        stride_length = (32, 32, 32)
        channel_size = 1

        reorient_template = ants.image_read(get_antsxnet_data("oasis"))
        reorient_template = pad_or_crop_image_to_size( reorient_template, image_size)

        center_of_mass_template = np.asarray(ants.get_center_of_mass(reorient_template))
        center_of_mass_image = np.asarray(ants.get_center_of_mass(image))
        translation = center_of_mass_image - center_of_mass_template
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)

        image_reoriented = xfrm.apply_to_image(image, reorient_template, interpolation="linear")
        roi_mask_reoriented = xfrm.apply_to_image(roi_mask, reorient_template, interpolation="nearestneighbor")
        roi_mask_reoriented = ants.threshold_image(roi_mask_reoriented, 0, 0, 0, 1)
        roi_inverted_mask_reoriented = ants.threshold_image(roi_mask_reoriented, 0, 0, 1, 0)

        inpainting_unet = create_partial_convolution_unet_model_3d((*patch_size, channel_size),
                                                                    number_of_priors=0,
                                                                    number_of_filters=(32, 64, 128, 256, 256),
                                                                    kernel_size=3)

        weights_name = ''
        if modality == "T1" or modality == "t1":
            weights_name = "wholeHeadInpaintingPatchBasedT1"
        elif modality == "FLAIR" or modality == "flair":
            weights_name = "wholeHeadInpaintingPatchBasedFLAIR"
        else:
            raise ValueError("Unavailable modality given: " + modality)

        weights_file_name = get_pretrained_network(weights_name,
            antsxnet_cache_directory=antsxnet_cache_directory)
        inpainting_unet.load_weights(weights_file_name)

        if verbose:
            print("Preprocessing:  Extracting patches.")

        image_patches = extract_image_patches(image_reoriented, patch_size, max_number_of_patches="all",
            stride_length=stride_length, random_seed=None, return_as_array=True)

        min_image_val = image_reoriented[roi_inverted_mask_reoriented == 1].min()
        max_image_val = image_reoriented[roi_inverted_mask_reoriented == 1].max()
        image_reoriented = (image_reoriented - min_image_val) / (max_image_val - min_image_val)

        image_patches_rescaled = extract_image_patches(image_reoriented, patch_size, max_number_of_patches="all",
            stride_length=stride_length, random_seed=None, return_as_array=True)
        mask_patches = extract_image_patches(roi_inverted_mask_reoriented, patch_size, max_number_of_patches="all",
            stride_length=stride_length, random_seed=None, return_as_array=True)

        batchX = np.expand_dims(image_patches_rescaled, axis=-1)
        batchXMask = np.expand_dims(mask_patches, axis=-1)

        batchX[batchXMask == 0] = 1

        predicted_data = np.zeros_like(batchX)

        for i in range(batchX.shape[0]):
            if np.any(batchXMask[i,:,:,:,:] == 0):
                if verbose:
                    print("  Predicting patch " + str(i) + " (of " + str(batchX.shape[0]) + ")")
                predicted_patch = inpainting_unet.predict([batchX[[i],:,:,:,:], batchXMask[[i],:,:,:,:]], verbose=verbose)
                predicted_patch_image = regression_match_image(ants.from_numpy(np.squeeze(predicted_patch)),
                                                               ants.from_numpy(np.squeeze(image_patches[i,:,:,:])),
                                                               mask=ants.from_numpy(np.squeeze(batchXMask[i,:,:,:,:]))
                                                               )
                predicted_data[i,:,:,:,0] = predicted_patch_image.numpy()
            else:
                predicted_data[i,:,:,:,:] = batchX[i,:,:,:,:]

        inpainted_image = reconstruct_image_from_patches(np.squeeze(predicted_data),
            image_reoriented, stride_length=stride_length)

        if verbose:
            print("Post-processing:  reorienting to original space.")

        xfrm_inv = xfrm.invert()
        inpainted_image = xfrm_inv.apply_to_image(inpainted_image, image, interpolation="linear")
        inpainted_image = ants.copy_image_info(image, inpainted_image)
        inpainted_image[roi_mask == 0] = image[roi_mask == 0]

        return(inpainted_image)
