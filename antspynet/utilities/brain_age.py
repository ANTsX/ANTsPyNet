import statistics
import numpy as np
import tensorflow.keras as keras

import ants

def brain_age(t1,
              do_preprocessing=True,
              number_of_simulations=0,
              sd_affine=0.01,
              antsxnet_cache_directory=None,
              verbose=False):

    """
    Estimate BrainAge from a T1-weighted MR image using the DeepBrainNet
    architecture and weights described here:

    https://github.com/vishnubashyam/DeepBrainNet

    and described in the following article:

    https://academic.oup.com/brain/article-abstract/doi/10.1093/brain/awaa160/5863667?redirectedFrom=fulltext

    Preprocessing on the training data consisted of:
       * n4 bias correction,
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
    List consisting of the segmentation image and probability images for
    each label.

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> deep = brain_age(image)
    >>> print("Predicted age: ", deep['predicted_age']
    """

    from ..utilities import preprocess_brain_image
    from ..utilities import get_pretrained_network
    from ..utilities import randomly_transform_image_data

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    ################################
    #
    # Preprocess images
    #
    ################################

    t1_preprocessed = t1
    if do_preprocessing == True:
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

    t1_preprocessed = (t1_preprocessed - t1_preprocessed.min()) / (t1_preprocessed.max() - t1_preprocessed.min())

    ################################
    #
    # Load model and weights
    #
    ################################

    model_weights_file_name = get_pretrained_network("brainAgeDeepBrainNet", antsxnet_cache_directory=antsxnet_cache_directory)
    model = keras.models.load_model(model_weights_file_name)

    # The paper only specifies that 80 slices are used for prediction.  I just picked
    # a reasonable range spanning the center of the brain

    which_slices = list(range(45, 125))

    batchX = np.zeros((len(which_slices), *t1_preprocessed.shape[0:2], 3))

    input_image = list()
    input_image.append(t1_preprocessed)

    input_image_list = list()
    input_image_list.append(input_image)

    if number_of_simulations > 0:
        data_augmentation = randomly_transform_image_data(
            reference_image=t1_preprocessed,
            input_image_list=input_image_list,
            number_of_simulations=number_of_simulations,
            transform_type='affine',
            sd_affine=sd_affine,
            input_image_interpolator='linear')

    brain_age_per_slice = None
    for i in range(number_of_simulations + 1):

        batch_image = t1_preprocessed
        if i > 0:
            batch_image = data_augmentation['simulated_images'][i-1][0]

        for j in range(len(which_slices)):

            slice = (ants.slice_image(batch_image, axis=2, idx=which_slices[j])).numpy()
            batchX[j,:,:,0] = slice
            batchX[j,:,:,1] = slice
            batchX[j,:,:,2] = slice

        if verbose == True:
            print("Brain age (DeepBrainNet):  predicting brain age per slice (batch = ", i, ")")

        if i == 0:
            brain_age_per_slice = model.predict(batchX, verbose=verbose)
        else:
            prediction = model.predict(batchX, verbose=verbose)
            brain_age_per_slice = brain_age_per_slice + (prediction - brain_age_per_slice) /  (i+1)

    predicted_age = statistics.median(brain_age_per_slice)[0]

    return_dict = {'predicted_age' : predicted_age,
                   'brain_age_per_slice' : brain_age_per_slice}
    return(return_dict)
