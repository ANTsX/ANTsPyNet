import numpy as np
import ants

def deep_atropos(t1,
                 do_preprocessing=True,
                 use_spatial_priors=1,
                 antsxnet_cache_directory=None,
                 verbose=False):

    """
    Six-tissue segmentation.

    Perform Atropos-style six tissue segmentation using deep learning.

    The labeling is as follows:
    Label 0 :  background
    Label 1 :  CSF
    Label 2 :  gray matter
    Label 3 :  white matter
    Label 4 :  deep gray matter
    Label 5 :  brain stem
    Label 6 :  cerebellum

    Preprocessing on the training data consisted of:
       * n4 bias correction,
       * denoising,
       * brain extraction, and
       * affine registration to MNI.
    The input T1 should undergo the same steps.  If the input T1 is the raw
    T1, these steps can be performed by the internal preprocessing, i.e. set
    do_preprocessing = True

    Arguments
    ---------
    t1 : ANTsImage
        raw or preprocessed 3-D T1-weighted brain image.

    do_preprocessing : boolean
        See description above.

    use_spatial_priors : integer
        Use MNI spatial tissue priors (0 or 1).  Currently, only '0' (no priors) and '1'
        (cerebellar prior only) are the only two options.  Default is 1.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> flash = deep_atropos(image)
    """

    from ..architectures import create_unet_model_3d
    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import preprocess_brain_image
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches
    from ..utilities import brain_extraction

    if not isinstance(t1, list):

        if t1.dimension != 3:
            raise ValueError("Image dimension must be 3.")

        ################################
        #
        # Preprocess images
        #
        ################################

        t1_preprocessed = t1
        if do_preprocessing:
            t1_preprocessing = preprocess_brain_image(t1,
                truncate_intensity=(0.01, 0.99),
                brain_extraction_modality="t1",
                template="croppedMni152",
                template_transform_type="antsRegistrationSyNQuickRepro[a]",
                do_bias_correction=True,
                do_denoising=True,
                antsxnet_cache_directory=antsxnet_cache_directory,
                verbose=verbose)
            t1_preprocessed = t1_preprocessing["preprocessed_image"] * t1_preprocessing['brain_mask']

        ################################
        #
        # Build model and load weights
        #
        ################################

        patch_size = (112, 112, 112)
        stride_length = (t1_preprocessed.shape[0] - patch_size[0],
                         t1_preprocessed.shape[1] - patch_size[1],
                         t1_preprocessed.shape[2] - patch_size[2])

        classes = ("background", "csf", "gray matter", "white matter",
                "deep gray matter", "brain stem", "cerebellum")

        mni_priors = None
        channel_size = 1
        if use_spatial_priors != 0:
            mni_priors = ants.ndimage_to_list(ants.image_read(get_antsxnet_data("croppedMni152Priors", antsxnet_cache_directory=antsxnet_cache_directory)))
            for i in range(len(mni_priors)):
                mni_priors[i] = ants.copy_image_info(t1_preprocessed, mni_priors[i])
            channel_size = 2

        unet_model = create_unet_model_3d((*patch_size, channel_size),
            number_of_outputs=len(classes), mode = "classification",
            number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            weight_decay=1e-5, additional_options=("attentionGating"))

        if verbose:
            print("DeepAtropos:  retrieving model weights.")

        weights_file_name = ''
        if use_spatial_priors == 0:
            weights_file_name = get_pretrained_network("sixTissueOctantBrainSegmentation", antsxnet_cache_directory=antsxnet_cache_directory)
        elif use_spatial_priors == 1:
            weights_file_name = get_pretrained_network("sixTissueOctantBrainSegmentationWithPriors1", antsxnet_cache_directory=antsxnet_cache_directory)
        else:
            raise ValueError("use_spatial_priors must be a 0 or 1")
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction.")

        t1_preprocessed = (t1_preprocessed - t1_preprocessed.mean()) / t1_preprocessed.std()
        image_patches = extract_image_patches(t1_preprocessed, patch_size=patch_size,
                                            max_number_of_patches="all", stride_length=stride_length,
                                            return_as_array=True)
        batchX = np.zeros((*image_patches.shape, channel_size))
        batchX[:,:,:,:,0] = image_patches
        if channel_size > 1:
            prior_patches = extract_image_patches(mni_priors[6], patch_size=patch_size,
                                max_number_of_patches="all", stride_length=stride_length,
                                return_as_array=True)
            batchX[:,:,:,:,1] = prior_patches

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        probability_images = list()
        for i in range(len(classes)):
            if verbose:
                print("Reconstructing image", classes[i])
            reconstructed_image = reconstruct_image_from_patches(predicted_data[:,:,:,:,i],
                domain_image=t1_preprocessed, stride_length=stride_length)

            if do_preprocessing:
                probability_images.append(ants.apply_transforms(fixed=t1,
                    moving=reconstructed_image,
                    transformlist=t1_preprocessing['template_transforms']['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose))
            else:
                probability_images.append(reconstructed_image)

        image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

        return_dict = {'segmentation_image' : segmentation_image,
                    'probability_images' : probability_images}
        return(return_dict)

    else:

        ################################
        #
        # Preprocess images
        #
        ################################

        hcp_t1_template = ants.image_read(get_antsxnet_data("hcpyaT1Template"))
        hcp_t2_template = ants.image_read(get_antsxnet_data("hcpyaT2Template"))
        hcp_fa_template = ants.image_read(get_antsxnet_data("hcpyaFATemplate"))
        hcp_template_brain_mask = ants.image_read(get_antsxnet_data("hcpyaTemplateBrainMask"))
        hcp_template_brain_segmentation = ants.image_read(get_antsxnet_data("hcpyaTemplateBrainSegmentation"))

        hcp_templates = list()
        hcp_templates.append(hcp_t1_template * hcp_template_brain_mask)
        hcp_templates.append(hcp_t2_template * hcp_template_brain_mask)
        hcp_templates.append(hcp_fa_template * hcp_template_brain_mask)

        reg = None
        t1_mask = None
        preprocessed_images = list()
        for i in range(len(t1)):
            n4 = ants.n4_bias_field_correction(t1[i], mask=t1[i]*0+1, 
                                               convergence={'iters': [50, 50, 50, 50], 'tol': 0.0},
                                               rescale_intensities=True, 
                                               verbose=verbose)
            if i == 0:            
                t1_mask = brain_extraction(t1[0], modality="t1", 
                                           antsxnet_cache_directory=antsxnet_cache_directory, 
                                           verbose=verbose)
                n4 = n4 * t1_mask
                # n4 = ants.histogram_match_image2(n4, hcp_templates[i],
                #                                  source_mask=t1_mask,
                #                                  reference_mask=hcp_template_brain_mask)
                reg = ants.registration(hcp_templates[i], n4, 
                                        type_of_transform="antsRegistrationSyNQuick[a]",
                                        verbose=verbose)
                preprocessed_images.append(reg['warpedmovout'])
            else:
                n4 = n4 * t1_mask
                # n4 = ants.histogram_match_image2(n4, hcp_templates[i],
                #                                  source_mask=t1_mask,
                #                                  reference_mask=hcp_template_brain_mask)
                n4 = ants.apply_transforms(hcp_templates[i], n4, 
                                           transformlist=reg['fwdtransforms'],
                                           verbose=verbose)
                preprocessed_images.append(n4)
                
            preprocessed_images[i] = ants.iMath_normalize(preprocessed_images[i])
        

        ################################
        #
        # Build model and load weights
        #
        ################################

        patch_size = (192, 224, 192)
        stride_length = (hcp_templates[0].shape[0] - patch_size[0],
                         hcp_templates[0].shape[1] - patch_size[1],
                         hcp_templates[0].shape[2] - patch_size[2])

        hcp_template_priors = list()
        for i in range(6):
            prior = ants.threshold_image(hcp_template_brain_segmentation, i+1, i+1, 1, 0)
            prior_smooth = ants.smooth_image(prior, 1.0)
            hcp_template_priors.append(prior_smooth)

        classes = ("background", "csf", "gray matter", "white matter",
                "deep gray matter", "brain stem", "cerebellum")
        number_of_classification_labels = len(classes)
        image_modalities = ["T1", "T2", "FA"]
        channel_size = len(image_modalities) + len(hcp_template_priors)

        unet_model = create_unet_model_3d((*patch_size, channel_size),
            number_of_outputs=number_of_classification_labels, mode="classification",
            number_of_filters=(16, 32, 64, 128), dropout_rate=0.0,
            convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
            weight_decay=0.0)

        if verbose:
            print("DeepAtropos:  retrieving model weights.")

        weights_file_name = get_pretrained_network("test_hcpya_da", 
                                                   antsxnet_cache_directory=antsxnet_cache_directory)
        unet_model.load_weights(weights_file_name)

        ################################
        #
        # Do prediction and normalize to native space
        #
        ################################

        if verbose:
            print("Prediction.")

        batchX = np.zeros((8, *patch_size, channel_size))
        
        image_patches_list = list()
        for i in range(len(preprocessed_images)):
            image_patches = extract_image_patches(preprocessed_images[i], 
                                                  patch_size=patch_size,
                                                  max_number_of_patches="all", 
                                                  stride_length=stride_length,
                                                  return_as_array=True)
            image_patches_list.append(image_patches)

        for i in range(len(preprocessed_images)):
            for j in range(8):
                batchX[j,:,:,:,i] = image_patches_list[i][j,:,:,:]

        priors_patches_list = list()
        for i in range(len(hcp_template_priors)):
            prior_patches = extract_image_patches(hcp_template_priors[i], 
                                                  patch_size=patch_size,
                                                  max_number_of_patches="all", 
                                                  stride_length=stride_length,
                                                  return_as_array=True)
            priors_patches_list.append(prior_patches)

        for i in range(len(hcp_template_priors)):
            for j in range(8):
                batchX[j,:,:,:,len(preprocessed_images) + i] = priors_patches_list[i][j,:,:,:]

        predicted_data = unet_model.predict(batchX, verbose=verbose)

        probability_images = list()
        for i in range(len(classes)):
            if verbose:
                print("Reconstructing image", classes[i])
            reconstructed_image = reconstruct_image_from_patches(predicted_data[:,:,:,:,i],
                domain_image=hcp_templates[0], stride_length=stride_length)

            if do_preprocessing:
                probability_images.append(ants.apply_transforms(fixed=t1[0],
                    moving=reconstructed_image,
                    transformlist=reg['invtransforms'],
                    whichtoinvert=[True], interpolator="linear", verbose=verbose))
            else:
                probability_images.append(reconstructed_image)

        image_matrix = ants.image_list_to_matrix(probability_images, t1[0] * 0 + 1)
        segmentation_matrix = np.argmax(image_matrix, axis=0)
        segmentation_image = ants.matrix_to_images(
            np.expand_dims(segmentation_matrix, axis=0), t1[0] * 0 + 1)[0]

        return_dict = {'segmentation_image' : segmentation_image,
                       'probability_images' : probability_images}
        return(return_dict)
        
            
        
        