import ants

import numpy as np
import pandas as pd

import tensorflow as tf

def check_xray_lung_orientation(image,
                                antsxnet_cache_directory=None,
                                verbose=False):

    """

    Check the correctness of image orientation, i.e., flipped left-right, up-down, 
    or both.  If True, attempts to correct before returning corrected image.  Otherwise
    it returns None.

    Arguments
    ---------
    image : ANTsImage
        raw 3-D MRI whole head image.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.
          
    verbose : boolean
        Print progress to the screen.

    Returns
    -------

    Best guess estimate of correctly oriented image or None if it's already correctly
    oriented.

    Example
    -------

    """

    from ..utilities import get_pretrained_network
    from ..architectures import create_resnet_model_2d

    if image.dimension != 2:
        raise ValueError( "Image dimension must be 2." )

    resampled_image_size = (224, 224)
    if image.shape != resampled_image_size:
        if verbose:
            print("Resampling image to", resampled_image_size)
        resampled_image = ants.resample_image(image, resampled_image_size, use_voxels=True)        

    model = create_resnet_model_2d((None, None, 1),
                                    number_of_classification_labels=3,
                                    mode="classification",
                                    layers=(1, 2, 3, 4),
                                    residual_block_schedule=(2, 2, 2, 2), lowest_resolution=64,
                                    cardinality=1, squeeze_and_excite=False)
    weights_file_name = get_pretrained_network("xrayLungOrientation",
                                                antsxnet_cache_directory=antsxnet_cache_directory)
    model.load_weights(weights_file_name)

    image_min = resampled_image.min()
    image_max = resampled_image.max()
    normalized_image = ants.image_clone(resampled_image)
    normalized_image = (normalized_image - image_min) / (image_max - image_min)
    batchX = np.expand_dims(normalized_image.numpy(), 0)
    batchX = np.expand_dims(batchX, -1)
    batchY = model.predict(batchX, verbose=verbose)

    # batchY is a 3-element array:
    #   batchY[0] = Pr(image is correctly oriented)
    #   batchY[1] = Pr(image is flipped up/down)
    #   batchY[2] = Pr(image is flipped left/right)        

    if batchY[0][0] > 0.5:
        return None
    else:
        if verbose:
            print("Possible incorrect orientation.  Attempting to correct.")
        image_up_down = ants.from_numpy(np.fliplr(normalized_image.numpy()), 
                                        origin=resampled_image.origin, 
                                        spacing=resampled_image.spacing, 
                                        direction=resampled_image.direction)
        image_left_right = ants.from_numpy(np.flipud(normalized_image.numpy()), 
                                           origin=resampled_image.origin, 
                                           spacing=resampled_image.spacing, 
                                           direction=resampled_image.direction)
        image_both = ants.from_numpy(np.fliplr(np.flipud(normalized_image.numpy())), 
                                     origin=resampled_image.origin, 
                                     spacing=resampled_image.spacing, 
                                     direction=resampled_image.direction)

        batchX = np.zeros((3, *resampled_image_size, 1))
        batchX[0,:,:,0] = image_up_down.numpy()
        batchX[1,:,:,0] = image_left_right.numpy()
        batchX[2,:,:,0] = image_both.numpy()
        batchY = model.predict(batchX, verbose=verbose)
        
        oriented_image = None
        if batchY[0][0] > batchY[1][0] and batchY[0][0] > batchY[2][0]:
            if verbose:
                print("Image is flipped up-down.")
            oriented_image = ants.from_numpy(np.fliplr(image.numpy()), 
                                             origin=image.origin, 
                                             spacing=image.spacing, 
                                             direction=image.direction)
        elif batchY[1][0] > batchY[0][0] and batchY[1][0] > batchY[2][0]:    
            if verbose:
                print("Image is flipped left-right.")
            oriented_image = ants.from_numpy(np.flipud(image.numpy()), 
                                             origin=image.origin, 
                                             spacing=image.spacing, 
                                             direction=image.direction)
        else:    
            if verbose:
                print("Image is flipped up-down and left-right.")
            oriented_image = ants.from_numpy(np.fliplr(np.flipud(image.numpy())), 
                                             origin=image.origin, 
                                             spacing=image.spacing, 
                                             direction=image.direction)
        return oriented_image         


def chexnet(image,
            check_image_orientation=False,
            antsxnet_cache_directory=None,
            verbose=False):

    """
    ANTsXNet reproduction of https://arxiv.org/pdf/1711.05225.pdf.  This includes
    our own network architecture and training (including data augmentation).
    
    "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning"

    Includes a network for checking the correctness of image orientation, i.e.,
    flipped left-right, up-down, or both.
    
    Disease categories:
       Atelectasis
       Cardiomegaly
       Consolidation
       Edema
       Effusion
       Emphysema
       Fibrosis 
       Hernia
       Infiltration
       Mass
       No Finding
       Nodule
       Pleural Thickening
       Pneumonia
       Pneumothorax

    Arguments
    ---------
    image : ANTsImage
        raw 3-D MRI whole head image.

    check_image_orientation : boolean
        Check the correctness of image orientation, i.e., flipped left-right, up-down, 
        or both.  If True, attempts to correct before prediction.
         
    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be reused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.
          
    verbose : boolean
        Print progress to the screen.

    Returns
    -------

    Data frame with prediction probability values for each disease category.

    Example
    -------
    """

    from ..utilities import get_pretrained_network
    from ..architectures import create_resnet_model_2d

    if image.dimension != 2:
        raise ValueError( "Image dimension must be 2." )

    ################################
    #
    # Resample to image size
    #
    ################################

    image_size = (224, 224)

    if image.shape != image_size:
        if verbose:
            print("Resampling image to", image_size)
        image = ants.resample_image(image, image_size, use_voxels=True)        

    if check_image_orientation:
        image = check_xray_lung_orientation(image)
            
    ################################
    #
    # Load model and weights
    #
    ################################

    weights_file_name = get_pretrained_network("chexnetClassification",
                                               antsxnet_cache_directory=antsxnet_cache_directory)

    disease_categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'NoFinding', 
                          'Nodule', 'PleuralThickening', 'Pneumonia', 'Pneumothorax']
    number_of_classification_labels = len(disease_categories)
    channel_size = 1

    model = create_resnet_model_2d((None, None, channel_size),
                                   number_of_classification_labels=number_of_classification_labels,
                                   mode="regression",
                                   layers=(1, 2, 3, 4),
                                   residual_block_schedule=(2, 2, 2, 2),
                                   lowest_resolution=64,
                                   cardinality=1,
                                   squeeze_and_excite=False)
    model.load_weights(weights_file_name)

    image = (image - image.mean()) / image.std()
    batchX = np.expand_dims(image.numpy(), 0)
    batchX = np.expand_dims(batchX, -1)

    batchY = (tf.nn.sigmoid(model.predict(batchX, verbose=verbose))).numpy()

    disease_category_df = pd.DataFrame(batchY, columns=disease_categories)

    return disease_category_df

