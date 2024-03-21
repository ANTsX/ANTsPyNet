import numpy as np
import ants
import warnings

def mouse_brain_extraction(image,
                           modality="t2",
                           return_isotropic_output=False,
                           which_axis=2,
                           antsxnet_cache_directory=None,
                           verbose=False):

    """
    Perform brain extraction of mouse MRI

    Arguments
    ---------
    image : ANTsImage
        input image

    modality : string
        "t2", "ex5coronal", "ex5sagittal".  The latter are E13.5 and E15.5 mouse 
        embroyonic histology data.
        
    return_isotropic_output : boolean
        The network actually learns an interpolating function specific to the 
        mouse brain.  Setting this to true, the output images are returned 
        isotropically resampled.

    which_axis : integer
        Specify direction for ex5 modalities.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = mouse_mri_brain_extraction(mri_image)
    """

    from ..architectures import create_unet_model_2d
    from ..architectures import create_unet_model_3d    
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    if modality == "t2": 

        template_shape = (176, 176, 176) 

        template = ants.image_read(get_antsxnet_data("bsplineT2MouseTemplate"))
        template = ants.resample_image(template, template_shape, use_voxels=True, interp_type=0)
        template_mask = ants.image_read(get_antsxnet_data("bsplineT2MouseTemplateBrainMask"))
        template_mask = ants.resample_image(template_mask, template_shape, use_voxels=True, interp_type=1)
    
        if verbose:
            print("Preprocessing:  Warping to B-spline T2w mouse template.")

        center_of_mass_reference = ants.get_center_of_mass(template_mask)
        center_of_mass_image = ants.get_center_of_mass(image)
        translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
        xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
            center=np.asarray(center_of_mass_reference), translation=translation)
        xfrm_inv = ants.invert_ants_transform(xfrm)
        
        image_warped = ants.apply_ants_transform_to_image(xfrm, image, template_mask, interpolation="linear")
        image_warped = (image_warped - image_warped.min()) / (image_warped.max() - image_warped.min())
        
        unet_model = create_unet_model_3d((*template_shape, 1),
                                           number_of_outputs=1, mode="sigmoid", 
                                           number_of_filters=(16, 32, 64, 128),
                                           convolution_kernel_size=(3, 3, 3), 
                                           deconvolution_kernel_size=(2, 2, 2))
        weights_file_name = get_pretrained_network("mouseT2wBrainExtraction3D")
        unet_model.load_weights(weights_file_name)
        
        batchX = np.zeros((1, *template_shape, 1))
        batchX[0,:,:,:,0] = image_warped.numpy()

        if verbose:
            print("Prediction.")
        batchY = np.squeeze(unet_model.predict(batchX, verbose=verbose))
        
        probability_mask = ants.from_numpy(batchY, origin=image_warped.origin,
                                           spacing=image_warped.spacing, direction=image_warped.direction)
        reference_image = image
        if return_isotropic_output:
            new_spacing = [np.array(image.spacing).min()] * len(image.spacing)
            reference_image = ants.resample_image(image, new_spacing, use_voxels=False, interp_type=0)

        probability_mask =  ants.apply_ants_transform_to_image(xfrm_inv, probability_mask, 
                                                               reference_image, interpolation="linear")

        return probability_mask

    # elif modality == "t2coronal":
        
    #     weights_file_name = get_pretrained_network("mouseMriBrainExtraction",
    #         antsxnet_cache_directory=antsxnet_cache_directory)

    #     resampled_image_size = (256, 256)
    #     original_slice_shape = image.shape
    #     if image.dimension > 2:
    #         original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    #     unet_model = create_unet_model_2d((*resampled_image_size, 1),
    #         number_of_outputs=1, mode="sigmoid",
    #         number_of_filters=(32, 64, 128, 256, 512),
    #         convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
    #         dropout_rate=0.0, weight_decay=0,
    #         additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    #     unet_model.load_weights(weights_file_name)

    #     if verbose:
    #         print("Preprocessing:  Resampling.")

    #     number_of_slices = 1
    #     if image.dimension > 2:
    #         number_of_slices = image.shape[which_axis]

    #     batch_X = np.zeros((number_of_slices, *resampled_image_size, 1))
        
    #     # Spacing is based on training data
    #     template = ants.from_numpy(np.zeros(resampled_image_size),
    #                                origin=(0, 0), spacing=(0.08, 0.08), direction=np.eye(2))

    #     count = 0
    #     xfrms = list()
    #     for j in range(number_of_slices):
    #         slice = None
    #         if image.dimension > 2:
    #             slice = ants.slice_image(image, axis=which_axis, idx=j, collapse_strategy=1)                
    #         else:
    #             slice = image
    #         if slice.max() > slice.min():
    #             center_of_mass_reference = ants.get_center_of_mass(template * 0 + 1)
    #             center_of_mass_image = ants.get_center_of_mass(slice)
    #             translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    #             xfrm = ants.create_ants_transform(transform_type="Euler2DTransform",
    #                 center=np.asarray(center_of_mass_reference), translation=translation)
    #             xfrms.append(xfrm)
    #             slice_resampled = ants.apply_ants_transform_to_image(xfrm, slice, template, interpolation="linear")
    #             slice_array = slice_resampled.numpy()
    #             slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
    #             batch_X[count,:,:,0] = slice_array
    #         else:
    #             xfrms.append(None)    
    #         count = count + 1

    #     if verbose:
    #         print("Prediction: ")

    #     predicted_data = unet_model.predict(batch_X, verbose=int(verbose))

    #     if verbose:
    #         print("Post-processing:  resampling to original space.")

    #     foreground_probability_array = np.zeros(image.shape)
    #     for j in range(number_of_slices):
    #         if xfrms[j] is None:
    #             continue
    #         reference_slice = ants.slice_image(image, axis=which_axis, idx=j, collapse_strategy=1) 
    #         slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:]),
    #                                           origin=template.origin, spacing=template.spacing,
    #                                           direction=template.direction)
    #         xfrm_inv = ants.invert_ants_transform(xfrms[j])
    #         slice = ants.apply_ants_transform_to_image(xfrm_inv, slice_resampled, reference_slice, interpolation="linear")                         
    #         if image.dimension == 2:
    #             foreground_probability_array[:,:] = slice.numpy()
    #         else:
    #             if which_axis == 0:
    #                 foreground_probability_array[j,:,:] = slice.numpy()
    #             elif which_axis == 1:
    #                 foreground_probability_array[:,j,:] = slice.numpy()
    #             else:
    #                 foreground_probability_array[:,:,j] = slice.numpy()

    #     origin = image.origin
    #     spacing = image.spacing
    #     direction = image.direction

    #     foreground_probability_image = ants.from_numpy(foreground_probability_array,
    #         origin=origin, spacing=spacing, direction=direction)

    #     return(foreground_probability_image)
    
    elif "ex5" in modality:

        weights_file_name = ""
        if "coronal" in modality.lower():
            weights_file_name = get_pretrained_network("ex5_coronal_weights",
                antsxnet_cache_directory=antsxnet_cache_directory)
        elif "sagittal" in modality.lower():
            weights_file_name = get_pretrained_network("ex5_sagittal_weights",
                antsxnet_cache_directory=antsxnet_cache_directory)
        else:
            raise ValueError("Valid axis view  options are coronal and sagittal.")

        resampled_image_size = (512, 512)
        original_slice_shape = image.shape
        if image.dimension > 2:
            original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

        unet_model = create_unet_model_2d((*resampled_image_size, 1),
            number_of_outputs=2, mode="classification",
            number_of_filters=(64, 96, 128, 256, 512),
            convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
            dropout_rate=0.0, weight_decay=0,
            additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
        unet_model.load_weights(weights_file_name)

        if verbose:
            print("Preprocessing:  Resampling.")

        number_of_channels = image.components
        number_of_slices = 1
        if image.dimension > 2:
            number_of_slices = image.shape[which_axis]

        image_channels = list()
        if number_of_channels == 1:
            image_channels.append(image)
        else:
            image_channels = ants.split_channels(image)

        batch_X = np.zeros((number_of_channels * number_of_slices, *resampled_image_size, 1))

        count = 0
        for i in range(number_of_channels):
            image_channel_array = image_channels[i].numpy()
            for j in range(number_of_slices):
                slice = None
                if image.dimension > 2:
                    if which_axis == 0:
                        image_channel_slice_array = np.squeeze(image_channel_array[j,:,:])
                    elif which_axis == 1:
                        image_channel_slice_array = np.squeeze(image_channel_array[:,j,:])
                    else:
                        image_channel_slice_array = np.squeeze(image_channel_array[:,:,j])
                    slice = ants.from_numpy(image_channel_slice_array)
                else:
                    slice = image_channels[i]
                if slice.max() > slice.min():
                    slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                    slice_array = slice_resampled.numpy()
                    slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
                    batch_X[count,:,:,0] = slice_array
                count = count + 1

        if verbose:
            print("Prediction: ")

        predicted_data = unet_model.predict(batch_X, verbose=int(verbose))
        if number_of_channels > 1:
            if verbose:
                print("Averaging across channels.")
            predicted_data_temp = np.split(predicted_data, number_of_channels, axis=0)
            predicted_data = np.zeros((number_of_slices, *resampled_image_size, 1))
            for i in range(number_of_channels):
                predicted_data = (predicted_data * i + predicted_data_temp[i]) / (i + 1)

        if verbose:
            print("Post-processing:  resampling to original space.")

        foreground_probability_array = np.zeros(image.shape)
        for j in range(number_of_slices):
            slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,1]))
            slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
            if image.dimension == 2:
                foreground_probability_array[:,:] = slice.numpy()
            else:
                if which_axis == 0:
                    foreground_probability_array[j,:,:] = slice.numpy()
                elif which_axis == 1:
                    foreground_probability_array[:,j,:] = slice.numpy()
                else:
                    foreground_probability_array[:,:,j] = slice.numpy()

        origin = image.origin
        spacing = image.spacing
        direction = image.direction

        foreground_probability_image = ants.from_numpy(foreground_probability_array,
            origin=origin, spacing=spacing, direction=direction)

        return(foreground_probability_image)
    
    else:  
        raise ValueError("Unrecognized type")

def mouse_brain_parcellation(image,
                             mask=None,
                             return_isotropic_output=False,
                             which_parcellation="nick",
                             antsxnet_cache_directory=None,
                             verbose=False):
    
    """
    Determine brain parcellation

    Arguments
    ---------
    image : ANTsImage
        input image based on "which" parcellation chosen.

    mask : ANTsImage
        Brain mask.  If not specified, one is estimated using ANTsXNet mouse brain 
        extraction.

    return_isotropic_output : boolean
        The network actually learns an interpolating function specific to the 
        mouse brain.  Setting this to true, the output images are returned 
        isotropically resampled.

    which_parcellation : string
        Brain parcellation type:
            * "nick" - t2w with labels:
                - 1: cerebral cortex
                - 2: cerebral nuclei
                - 3: brain stem
                - 4: cerebellum
                - 5: main olfactory bulb
                - 6: hippocampal formation
                
    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Segmentation image and corresponding probability images.

    Example
    -------
    >>> output = mouse_brain_parcellation(image, type="nick")
    """
      
    from ..architectures import create_unet_model_3d    
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size

    if which_parcellation == "nick": 

        template_spacing = (0.075, 0.075, 0.075)
        template_crop_size = (176, 128, 240)
        
        template = ants.image_read(get_antsxnet_data("DevCCF_P56_MRI-T2_50um"))
        ants.set_spacing(template, (0.05, 0.05, 0.05))
        template = ants.resample_image(template, template_spacing, use_voxels=False, interp_type=4)
        template = pad_or_crop_image_to_size(template, template_crop_size)
        template_ri = ants.rank_intensity(template)

        template_mask = ants.image_read(get_antsxnet_data("DevCCF_P56_MRI-T2_50um_BrainParcellationNickMask"))
        ants.set_spacing(template_mask, (0.05, 0.05, 0.05))
        template_mask = ants.resample_image(template_mask, template_spacing, use_voxels=False, interp_type=1)
        template_mask = pad_or_crop_image_to_size(template_mask, template_crop_size)

        number_of_nonzero_labels = len(np.unique(template_mask.numpy())) - 1

        template_priors = list()
        for i in range(number_of_nonzero_labels):
            single_label = ants.threshold_image(template_mask, i+1, i+1)
            prior = ants.smooth_image(single_label, sigma=0.003, sigma_in_physical_coordinates=True)
            template_priors.append(prior)

        if mask is None:
            if verbose:
                print("Preprocessing:  Bias field correction and brain extraction.")

            mask = mouse_brain_extraction(image, modality="t2", 
                                          antsxnet_cache_directory=antsxnet_cache_directory, 
                                          verbose=verbose)   
            mask = ants.threshold_image(mask, 0.5, 1.1, 1, 0)
            mask = ants.label_clusters(mask, fully_connected=True)
            mask = ants.threshold_image(mask, 0, 0, 0, 1)
            
        image_brain = image * mask    

        if verbose:
            print("Preprocessing:  Warping to DevCCF P56 T2w mouse template.")

        reg = ants.registration(template, image_brain, type_of_transform="antsRegistrationSyNQuick[a]", verbose=int(verbose))

        image_warped = ants.rank_intensity(reg['warpedmovout'])
        image_warped = ants.histogram_match_image(image_warped, template_ri)
        image_warped = (image_warped - image_warped.min()) / (image_warped.max() - image_warped.min())

        number_of_filters = (16, 32, 64, 128, 256)
        number_of_classification_labels = number_of_nonzero_labels + 1
        channel_size = 1 + number_of_nonzero_labels
         
        unet_model = create_unet_model_3d((*template.shape, channel_size),
                        number_of_outputs=number_of_classification_labels, 
                        mode="classification", 
                        number_of_filters=number_of_filters,
                        convolution_kernel_size=(3, 3, 3), 
                        deconvolution_kernel_size=(2, 2, 2))
        weights_file_name = get_pretrained_network("mouseT2wBrainParcellation3DNick")
        unet_model.load_weights(weights_file_name)
        
        batchX = np.zeros((1, *template.shape, channel_size))
        batchX[0,:,:,:,0] = image_warped.numpy()
        for i in range(len(template_priors)):
            batchX[0,:,:,:,i+1] = template_priors[i].numpy()
            
        if verbose:
            print("Prediction.")
        batchY = unet_model.predict(batchX, verbose=verbose)

        reference_image = image
        if return_isotropic_output:
            new_spacing = [np.array(image.spacing).min()] * len(image.spacing)
            reference_image = ants.resample_image(image, new_spacing, use_voxels=False, interp_type=0)

        probability_images = list()
        for i in range(number_of_classification_labels):
            if verbose:
                print("Reconstructing image ", str(i))
            probability_image = ants.from_numpy(np.squeeze(batchY[0,:,:,:,i]), origin=template.origin,
                                                spacing=template.spacing, direction=template.direction)    
            probability_images.append(ants.apply_transforms(fixed=reference_image,
                moving=probability_image, transformlist=reg['invtransforms'],
                whichtoinvert=[True], interpolator="linear", verbose=verbose))

        image_matrix = ants.image_list_to_matrix(probability_images, reference_image * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), reference_image * 0 + 1)[0]

        return_dict = {'segmentation_image' : segmentation_image,
                       'probability_images' : probability_images}
        return(return_dict)
    
    else:  
        raise ValueError("Unrecognized parcellation.")


def mouse_cortical_thickness(t2,
                             mask=None,
                             return_isotropic_output=False,
                             antsxnet_cache_directory=None,
                             verbose=False):

    """
    Perform KellyKapowski cortical thickness using mouse_brain_parcellation 
    for segmentation.  Description concerning implementaiton and evaluation:

    https://www.medrxiv.org/content/10.1101/2020.10.19.20215392v1

    Arguments
    ---------
    t2 : ANTsImage
        input 3-D unprocessed T2-weighted whole mouse brain image.

    mask : ANTsImage
        Brain mask.  If not specified, one is estimated using ANTsXNet mouse brain 
        extraction.

    return_isotropic_output : boolean
        The underling parcellation network actually learns an interpolating function 
        specific to the mouse brain which is used for computing the cortical thickness 
        image.  Setting this to true, the returned output images are in this isotropically 
        resampled space.  Otherwise, they are in the sampled space of the input image.
        
    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Cortical thickness image and segmentation probability images.

    Example
    -------
    >>> image = ants.image_read("t2w_image.nii.gz")
    >>> kk = mouse_cortical_thickness(image)
    """

    if t2.dimension != 3:
        raise ValueError("Image dimension must be 3.")

    parcellation = mouse_brain_parcellation(t2, mask=mask,
                                            which_parcellation="nick",      
                                            return_isotropic_output=True,                                    
                                            antsxnet_cache_directory=antsxnet_cache_directory, 
                                            verbose=verbose)

    # Kelly Kapowski cortical thickness

    kk_segmentation = ants.image_clone(parcellation['segmentation_image'])
    kk_segmentation[kk_segmentation == 2] = 3
    kk_segmentation[kk_segmentation == 1] = 2
    kk_segmentation[kk_segmentation == 6] = 2
    cortical_matter = parcellation['probability_images'][1] + parcellation['probability_images'][6]
    other_matter = parcellation['probability_images'][2] + parcellation['probability_images'][3]

    kk = ants.kelly_kapowski(s=kk_segmentation, g=cortical_matter, w=other_matter,
                            its=45, r=0.0025, m=1.5, x=0, t=10,
                            verbose=int(verbose))

    if not return_isotropic_output:
        kk = ants.resample_image(kk, t2.spacing, use_voxels=False, interp_type=0)
        parcellation['segmentation_image'] = ants.resample_image(parcellation['segmentation_image'], 
                                                                 t2.spacing, use_voxels=False, interp_type=1)
        for i in range(len(parcellation['probability_images'])):
            parcellation['probability_images'][i] = ants.resample_image(parcellation['probability_images'][i], 
                                                                    t2.spacing, use_voxels=False, interp_type=0)

    return_dict = {'thickness_image' : kk,
                   'parcellation' : parcellation
                  }
    return(return_dict)


def mouse_histology_brain_mask(image,
                               which_axis=2,
                               antsxnet_cache_directory=None,
                               verbose=False):

    """
    Determine brain foreground of mouse data.

    Arguments
    ---------
    image : ANTsImage
        input image

    which_axis : integer
        If 3-D image, which_axis specifies the direction of the "view".

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = mouse_histology_brain_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = get_pretrained_network("allen_brain_mask_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=2, mode="classification",
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        dropout_rate=0.0, weight_decay=0,
        additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    unet_model.load_weights(weights_file_name)

    if verbose:
        print("Preprocessing:  Resampling.")

    number_of_channels = image.components
    number_of_slices = 1
    if image.dimension > 2:
        number_of_slices = image.shape[which_axis]

    image_channels = list()
    if number_of_channels == 1:
        image_channels.append(image)
    else:
        image_channels = ants.split_channels(image)

    batch_X = np.zeros((number_of_channels * number_of_slices, *resampled_image_size, 1))

    count = 0
    for i in range(number_of_channels):
        image_channel_array = image_channels[i].numpy()
        for j in range(number_of_slices):
            slice = None
            if image.dimension > 2:
                if which_axis == 0:
                    image_channel_slice_array = np.squeeze(image_channel_array[j,:,:])
                elif which_axis == 1:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,j,:])
                else:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,:,j])
                slice = ants.from_numpy(image_channel_slice_array)
            else:
                slice = image_channels[i]
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_array = slice_resampled.numpy()
                slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
                batch_X[count,:,:,0] = slice_array
            count = count + 1

    if verbose:
        print("Prediction: ")

    predicted_data = unet_model.predict(batch_X, verbose=int(verbose))
    if number_of_channels > 1:
        if verbose:
            print("Averaging across channels.")
        predicted_data_temp = np.split(predicted_data, number_of_channels, axis=0)
        predicted_data = np.zeros((number_of_slices, *resampled_image_size, 1))
        for i in range(number_of_channels):
            predicted_data = (predicted_data * i + predicted_data_temp[i]) / (i + 1)

    if verbose:
        print("Post-processing:  resampling to original space.")

    foreground_probability_array = np.zeros(image.shape)
    for j in range(number_of_slices):
        slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,1]))
        slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
        if image.dimension == 2:
            foreground_probability_array[:,:] = slice.numpy()
        else:
            if which_axis == 0:
                foreground_probability_array[j,:,:] = slice.numpy()
            elif which_axis == 1:
                foreground_probability_array[:,j,:] = slice.numpy()
            else:
                foreground_probability_array[:,:,j] = slice.numpy()

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    foreground_probability_image = ants.from_numpy(foreground_probability_array,
        origin=origin, spacing=spacing, direction=direction)

    return(foreground_probability_image)

def mouse_histology_hemispherical_coronal_mask(image,
                                               which_axis=2,
                                               antsxnet_cache_directory=None,
                                               verbose=False):

    """
    Determine left and right hemisphere brain masks of histology mouse data in coronal
    acquisitions for both P* and E*x5 data.  This assumes that the original histology
    image has been pre-extracted.

    Arguments
    ---------
    image : ANTsImage
        input image

    which_axis : integer
        If 3-D image, which_axis specifies the direction of the coronal view.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = mouse_histology_hemispherical_coronal_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = get_pretrained_network("allen_brain_leftright_coronal_mask_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    classes = (0, 1, 2)
    number_of_classification_labels = len(classes)

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=number_of_classification_labels, mode="classification",
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        dropout_rate=0.0, weight_decay=0,
        additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    unet_model.load_weights(weights_file_name)

    if verbose:
        print("Preprocessing:  Resampling.")

    number_of_channels = image.components
    number_of_slices = 1
    if image.dimension > 2:
        number_of_slices = image.shape[which_axis]

    image_channels = list()
    if number_of_channels == 1:
        image_channels.append(image)
    else:
        image_channels = ants.split_channels(image)

    batch_X = np.zeros((number_of_channels * number_of_slices, *resampled_image_size, 1))

    count = 0
    for i in range(number_of_channels):
        image_channel_array = image_channels[i].numpy()
        for j in range(number_of_slices):
            slice = None
            if image.dimension > 2:
                if which_axis == 0:
                    image_channel_slice_array = np.squeeze(image_channel_array[j,:,:])
                elif which_axis == 1:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,j,:])
                else:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,:,j])
                slice = ants.from_numpy(image_channel_slice_array)
            else:
                slice = image_channels[i]
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_smoothed = ants.smooth_image(slice_resampled, 1.0)
                slice_array = slice_smoothed.numpy()
                slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
                batch_X[count,:,:,0] = slice_array
            count = count + 1

    if verbose:
        print("Prediction: ")

    predicted_data = unet_model.predict(batch_X, verbose=int(verbose))
    if number_of_channels > 1:
        if verbose:
            print("Averaging across channels.")
        predicted_data_temp = np.split(predicted_data, number_of_channels, axis=0)
        predicted_data = np.zeros((number_of_slices, *resampled_image_size, 1))
        for i in range(number_of_channels):
            predicted_data = (predicted_data * i + predicted_data_temp[i]) / (i + 1)

    if verbose:
        print("Post-processing:  resampling to original space.")

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    probability_images = list()
    for i in range(len(classes)):
        if verbose == True:
            print("Reconstructing image", classes[i])

        probability_image_array = np.zeros(image.shape)
        for j in range(number_of_slices):
            slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,i]))
            slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
            if image.dimension == 2:
                probability_image_array[:,:] = slice.numpy()
            else:
                if which_axis == 0:
                    probability_image_array[j,:,:] = slice.numpy()
                elif which_axis == 1:
                    probability_image_array[:,j,:] = slice.numpy()
                else:
                    probability_image_array[:,:,j] = slice.numpy()

        probability_images.append(ants.from_numpy(probability_image_array,
            origin=origin, spacing=spacing, direction=direction))

    image_matrix = ants.image_list_to_matrix(probability_images, image_channels[0] * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), image_channels[0] * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return(return_dict)


def mouse_histology_cerebellum_mask(image,
                                    which_axis=2,
                                    view='sagittal',
                                    antsxnet_cache_directory=None,
                                    verbose=False):

    """
    Determine cerebellum foreground of mouse data.

    Arguments
    ---------
    image : ANTsImage
        input image

    which_axis : integer
        If 3-D image, which_axis specifies the direction of the "view".

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Foreground probability image.

    Example
    -------
    >>> output = mouse_histology_cerebellum_mask(histology_image)
    """

    from ..architectures import create_unet_model_2d
    from ..utilities import get_pretrained_network

    if which_axis < 0 or which_axis > 2:
        raise ValueError("Chosen axis not supported.")

    weights_file_name = None
    if view == "sagittal":
        weights_file_name = get_pretrained_network("allen_cerebellum_sagittal_mask_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    elif view == "coronal":
        weights_file_name = get_pretrained_network("allen_cerebellum_coronal_mask_weights",
            antsxnet_cache_directory=antsxnet_cache_directory)
    else:
        raise ValueError("Unrecognized option for view.  Must be sagittal or coronal.")

    resampled_image_size = (512, 512)
    original_slice_shape = image.shape
    if image.dimension > 2:
        original_slice_shape = tuple(np.delete(np.array(image.shape), which_axis))

    unet_model = create_unet_model_2d((*resampled_image_size, 1),
        number_of_outputs=1, mode="sigmoid",
        number_of_filters=(64, 96, 128, 256, 512),
        convolution_kernel_size=(3, 3), deconvolution_kernel_size=(2, 2),
        dropout_rate=0.0, weight_decay=0,
        additional_options=("initialConvolutionKernelSize[5]", "attentionGating"))
    unet_model.load_weights(weights_file_name)

    if verbose:
        print("Preprocessing:  Resampling.")

    number_of_channels = image.components
    number_of_slices = 1
    if image.dimension > 2:
        number_of_slices = image.shape[which_axis]

    image_channels = list()
    if number_of_channels == 1:
        image_channels.append(image)
    else:
        image_channels = ants.split_channels(image)

    batch_X = np.zeros((number_of_channels * number_of_slices, *resampled_image_size, 1))

    count = 0
    for i in range(number_of_channels):
        image_channel_array = image_channels[i].numpy()
        for j in range(number_of_slices):
            slice = None
            if image.dimension > 2:
                if which_axis == 0:
                    image_channel_slice_array = np.squeeze(image_channel_array[j,:,:])
                elif which_axis == 1:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,j,:])
                else:
                    image_channel_slice_array = np.squeeze(image_channel_array[:,:,j])
                slice = ants.from_numpy(image_channel_slice_array)
            else:
                slice = image_channels[i]
            if slice.max() > slice.min():
                slice_resampled = ants.resample_image(slice, resampled_image_size, use_voxels=True, interp_type=0)
                slice_array = slice_resampled.numpy()
                slice_array = (slice_array - slice_array.min()) / (slice_array.max() - slice_array.min())
                batch_X[count,:,:,0] = slice_array
            count = count + 1

    if verbose:
        print("Prediction: ")

    predicted_data = unet_model.predict(batch_X, verbose=int(verbose))
    if number_of_channels > 1:
        if verbose:
            print("Averaging across channels.")
        predicted_data_temp = np.split(predicted_data, number_of_channels, axis=0)
        predicted_data = np.zeros((number_of_slices, *resampled_image_size, 1))
        for i in range(number_of_channels):
            predicted_data = (predicted_data * i + predicted_data_temp[i]) / (i + 1)

    if verbose:
        print("Post-processing:  resampling to original space.")

    foreground_probability_array = np.zeros(image.shape)
    for j in range(number_of_slices):
        slice_resampled = ants.from_numpy(np.squeeze(predicted_data[j,:,:,0]))
        slice = ants.resample_image(slice_resampled, original_slice_shape, use_voxels=True, interp_type=0)
        if image.dimension == 2:
            foreground_probability_array[:,:] = slice.numpy()
        else:
            if which_axis == 0:
                foreground_probability_array[j,:,:] = slice.numpy()
            elif which_axis == 1:
                foreground_probability_array[:,j,:] = slice.numpy()
            else:
                foreground_probability_array[:,:,j] = slice.numpy()

    origin = image.origin
    spacing = image.spacing
    direction = image.direction

    foreground_probability_image = ants.from_numpy(foreground_probability_array,
        origin=origin, spacing=spacing, direction=direction)

    return(foreground_probability_image)

def mouse_histology_super_resolution(image,
                                     antsxnet_cache_directory=None,
                                     verbose=False):

    """
    Super resolution (2x) of a single image slice (256x256 -> 512x512)

    Arguments
    ---------
    image : ANTsImage or ANTsImage list
        input image or input image list

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Super resolution image of size 512x512 voxels (or a list depending on the input)

    Example
    -------
    >>> output = mouse_histology_super_resolution(histology_image)
    """

    from ..architectures import create_deep_back_projection_network_model_2d
    from ..utilities import get_pretrained_network
    from ..utilities import regression_match_image

    image_list = list()
    if isinstance(image, list):
        image_list = image
    else:
        image_list.append(image)

    lr_image_size = (256, 256)
    sr_image_size = (512, 512)

    image_lr_list = list()
    for i in range(len(image_list)):

        if image_list[i].components != 3:
            raise ValueError("Number of image channels should be 3 (rgb).")
        if image_list[i].dimension != 2:
            raise ValueError("Input image should be 2-D.")

        do_resample = False
        if image_list[i].shape != lr_image_size:
            warnings.warn("Resampling input image to (256, 256).")
            do_resample = True

        image_lr = ants.image_clone(image_list[i])
        image_lr_channels = ants.split_channels(image_lr)
        for c in range(len(image_lr_channels)):
            if do_resample:
                image_lr_channels[c] = ants.resample_image(image_lr_channels[c], resample_params=lr_image_size,
                                                           use_voxels=True, interp_type=0)
            image_lr_channels[c] = ((image_lr_channels[c] - image_lr_channels[c].min()) /
                                    (image_lr_channels[c].max() - image_lr_channels[c].min()))
        image_lr = ants.merge_channels(image_lr_channels)
        image_lr_list.append(image_lr)

    weights_file_name = get_pretrained_network("allen_sr_weights",
        antsxnet_cache_directory=antsxnet_cache_directory)

    sr_model = create_deep_back_projection_network_model_2d((*lr_image_size, 3),
        number_of_outputs=3, convolution_kernel_size=(6, 6), strides=(2, 2))
    sr_model.load_weights(weights_file_name)

    batch_X = np.zeros((len(image_lr_list), *lr_image_size, 3))
    for i in range(len(image_lr_list)):
        batch_X[i,:,:,:] = image_lr_list[i].numpy()

    if verbose:
        print("Prediction: ")

    predicted_data = sr_model.predict(batch_X, verbose=int(verbose))

    if verbose:
        print("Regression match output image.")

    spacing_factor = (lr_image_size[0] - 1) / (sr_image_size[0] - 1)

    image_sr_list = list()
    for i in range(len(image_lr_list)):
        image_sr = ants.from_numpy(predicted_data[i,:,:,:], origin=image_list[i].origin,
                                   direction=image_list[i].direction,
                                   spacing=(spacing_factor * image_list[i].spacing[0],
                                            spacing_factor * image_list[i].spacing[1]),
                                   has_components=True)
        image_channels = ants.split_channels(image_list[i])
        image_sr_channels = ants.split_channels(image_sr)
        for c in range(len(image_sr_channels)):
            image_lr_channel_resampled = ants.resample_image(image_channels[c], resample_params=sr_image_size,
                                                             use_voxels=True, interp_type=0)
            image_sr_channels[c] = regression_match_image(image_sr_channels[c], image_lr_channel_resampled)
        image_sr_list.append(ants.merge_channels(image_sr_channels))

    if isinstance(image, list):
        return(image_sr_list)
    else:
        return(image_sr_list[0])


