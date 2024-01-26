import ants
import numpy as np

def whole_head_inpainting(image,
                          roi_mask,
                          modality="t1",
                          mode="axial",
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
            
    mode : string
        Options include:
            * "sagittal": sagittal view network
            * "coronal": coronal view network
            * "axial": axial view network
            * "average": average of all canonical views
            * "meg": morphological erosion, greedy, iterative                

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
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

    from ..architectures import create_rmnet_generator
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if mode == "sagittal" or mode == "coronal" or mode == "axial":

        if verbose:
            print("Preprocessing:  Reorientation.")

        reorient_template = ants.image_read(get_antsxnet_data("nki"))
        reorient_template = pad_or_crop_image_to_size(reorient_template, (256, 256, 256))

        center_of_mass_template = np.round(np.asarray(ants.get_center_of_mass(reorient_template)))
        center_of_mass_image = np.round(np.asarray(ants.get_center_of_mass(image)))
        translation = center_of_mass_image - center_of_mass_template
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_template), translation=translation)

        image_reoriented = xfrm.apply_to_image(image, reorient_template, interpolation="linear")
        roi_mask_reoriented = xfrm.apply_to_image(roi_mask, reorient_template, interpolation="nearestneighbor")
        roi_mask_reoriented = ants.threshold_image(roi_mask_reoriented, 0, 0, 0, 1)

        geoms = ants.label_geometry_measures(roi_mask_reoriented)
        if geoms.shape[0] != 1:
            raise ValueError("ROI is not specified correctly.")

        lower_slice = None
        upper_slice = None
        weights_file = None
        direction = -1
        if mode == "sagittal":
            lower_slice = int(geoms['BoundingBoxLower_x'])
            upper_slice = int(geoms['BoundingBoxUpper_x'])

            raise ValueError("Weights not available yet.")

            weights_file = get_pretrained_network("inpainting_sagittal_rmnet_weights", 
                                                  antsxnet_cache_directory=antsxnet_cache_directory)
            direction = 0
        elif mode == "coronal":
            lower_slice = int(geoms['BoundingBoxLower_y'])
            upper_slice = int(geoms['BoundingBoxUpper_y'])
            
            raise ValueError("Weights not available yet.")

            weights_file = get_pretrained_network("inpainting_coronal_rmnet_weights", 
                                                  antsxnet_cache_directory=antsxnet_cache_directory)
            direction = 1
        elif mode == "axial":
            lower_slice = int(geoms['BoundingBoxLower_z'])
            upper_slice = int(geoms['BoundingBoxUpper_z'])
            weights_file = get_pretrained_network("inpainting_axial_rmnet_weights", 
                                                  antsxnet_cache_directory=antsxnet_cache_directory)
            direction = 2 

        model = create_rmnet_generator()
        model.load_weights(weights_file)

        number_of_slices = upper_slice - lower_slice + 1

        image_size = (256, 256)
        channel_size = 3
        batchX = np.zeros((number_of_slices, *image_size, channel_size))
        batchXMask = np.zeros((number_of_slices, *image_size, 1))
        batchX_max_values = np.zeros((number_of_slices,))         

        for i in range(number_of_slices):
            slice_index = i + lower_slice
            t1_slice = ants.slice_image(image_reoriented, axis=direction, 
                                        idx=slice_index, collapse_strategy=1)  
            batchX[i,:,:,0] = t1_slice.numpy()
            batchX_max_values[i] = batchX[i,:,:,0].max()
            batchX[i,:,:,0] = batchX[i,:,:,0] / (0.5 * batchX_max_values[i] ) - 1.
            for j in range(1, channel_size):
                batchX[i,:,:,j] = batchX[i,:,:,0]
            roi_mask_slice = ants.slice_image(roi_mask_reoriented, axis=direction,
                                              idx=slice_index, collapse_strategy=1)
            batchXMask[i,:,:,0] = roi_mask_slice.numpy()
                            
        batchY = model.predict([batchX, batchXMask], verbose=True)[:,:,:,0:3]
        
        inpainted_image_reoriented_array = image_reoriented.numpy()
        for i in range(number_of_slices):
            slice_idx = i + lower_slice
            inpainted_values = (np.mean(batchY[i,:,:,:], axis=-1) + 1) * (0.5 * batchX_max_values[i])
            if direction == 0:
                inpainted_image_reoriented_array[slice_idx,:,:] = inpainted_values
            elif direction == 1:
                inpainted_image_reoriented_array[:,slice_idx,:] = inpainted_values
            elif direction == 2:
                inpainted_image_reoriented_array[:,:,slice_idx] = inpainted_values
        inpainted_image_reoriented = ants.from_numpy(inpainted_image_reoriented_array) 
        inpainted_image_reoriented = ants.copy_image_info(image_reoriented, inpainted_image_reoriented)
                
        xfrm_inv = xfrm.invert()
        inpainted_image = xfrm_inv.apply_to_image(inpainted_image_reoriented, image, interpolation="linear")
        inpainted_image = ants.copy_image_info(image, inpainted_image)
        inpainted_image[roi_mask == 0] = image[roi_mask == 0]

        return(inpainted_image)
    
    elif mode == "average":
        
        sagittal = whole_head_inpainting(image, roi_mask=roi_mask, 
                                         modality=modality, mode="sagittal", 
                                         verbose=verbose) 
        coronal = whole_head_inpainting(image, roi_mask=roi_mask, 
                                        modality=modality, mode="coronal", 
                                        verbose=verbose) 
        axial = whole_head_inpainting(image, roi_mask=roi_mask, 
                                      modality=modality, mode="axial", 
                                      verbose=verbose) 
        
        return ((sagittal + coronal + axial)/3)

    elif mode == "meg":
        
        current_image = ants.image_clone(image)
        current_roi_mask = ants.threshold_image(roi_mask, 0, 0, 0, 1)
        roi_mask_volume = current_roi_mask.sum()

        iteration = 0
        while roi_mask_volume > 0:
            if verbose:
                print("roi_mask_volume (" + str(iteration) + "): " + str(roi_mask_volume)) 
                
            current_image = whole_head_inpainting(current_image, roi_mask=current_roi_mask, 
                                                  modality=modality, mode="average", 
                                                  verbose=verbose) 
            current_roi_mask = ants.iMath_ME(current_roi_mask, radius=1)
            roi_mask_volume = current_roi_mask.sum()
            iteration += 1
            
        return(current_image)
        
        



