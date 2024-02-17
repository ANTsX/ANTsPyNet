import ants

import numpy as np
import pandas as pd

def mri_modality_classification(image,
                                antsxnet_cache_directory=None,
                                verbose=False):

    """
    Predict MRI modality type (whole-head only).

    Modalities:
        T1
        T2
        FLAIR
        T2Star
        Mean DWI
        Mean Bold
        ASL perfusion

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

    Data frame with prediction probability values for each modality type.

    Example
    -------
    >>> image = ants.image_read(antspynet.get_antsxnet_data("mprage_hippmapp3r"))
    >>> classification = mri_modality_classification(image)
    """

    from ..utilities import get_pretrained_network
    from ..utilities import get_antsxnet_data
    from ..utilities import pad_or_crop_image_to_size
    from ..architectures import create_resnet_model_3d

    if image.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    ################################
    #
    # Normalize to template
    #
    ################################

    image_size = (112, 112, 112)
    resample_size = (2, 2, 2)

    template = ants.image_read(get_antsxnet_data("kirby"))
    template = ants.resample_image(template, resample_size)
    template = pad_or_crop_image_to_size(template, image_size)
    direction = template.direction
    direction[0, 0] = 1.0
    ants.set_direction(template, direction)
    ants.set_origin(template, (0, 0, 0))

    center_of_mass_template = ants.get_center_of_mass(template*0 + 1)
    center_of_mass_image = ants.get_center_of_mass(image*0 + 1)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)
    image = ants.apply_ants_transform_to_image(xfrm, image, template)

    image = (image - image.min()) / (image.max() - image.min())

    ################################
    #
    # Load model and weights
    #
    ################################

    weights_file_name = get_pretrained_network("mriModalityClassification",
                                               antsxnet_cache_directory=antsxnet_cache_directory)

    modality_types = ["T1", "T2", "FLAIR", "T2Star", "Mean DWI", "Mean Bold", "ASL Perfusion"]

    number_of_classification_labels = len(modality_types)
    channel_size = 1

    model = create_resnet_model_3d((None, None, None, channel_size),
                                   number_of_classification_labels=number_of_classification_labels,
                                   mode="classification",
                                   layers=(1, 2, 3, 4),
                                   residual_block_schedule=(3, 4, 6, 3),
                                   lowest_resolution=64,
                                   cardinality=1,
                                   squeeze_and_excite=False)
    model.load_weights(weights_file_name)

    batchX = np.expand_dims(image.numpy(), 0)
    batchX = np.expand_dims(batchX, -1)

    batchY = model.predict(batchX, verbose=verbose)

    modality_df = pd.DataFrame(batchY, columns=modality_types)

    return modality_df

