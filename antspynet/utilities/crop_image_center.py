import ants

import numpy as np
import math

def crop_image_center(image,
                      crop_size):
    """
    Crop the center of an image.

    Arguments
    ---------
    image : ANTsImage
        Input image

    crop_size: n-D tuple (depending on dimensionality).
        Width, height, depth (if 3-D), and time (if 4-D) of crop region.

    Returns
    -------
    A list (or array) of patches.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> cropped_image = crop_image_center(image, crop_size=(64, 64))
    """

    image_size = image.shape

    if len(image_size) != len(crop_size):
        raise ValueError("crop_size does not match image size.")

    if (np.asarray(crop_size) > np.asarray(image_size)).any():
        raise ValueError("A crop_size dimension is larger than image_size.")

    label_image = ants.image_clone(image) * 0
    start_index = (np.floor(0.5 * (np.asarray(image_size) - np.asarray(crop_size)))).astype(int)
    end_index = start_index + np.asarray(crop_size).astype(int)

    if image.dimension == 2:
        label_image[start_index[0]:end_index[0],
                    start_index[1]:end_index[1]] = 1
    elif image.dimension == 3:
        label_image[start_index[0]:end_index[0],
                    start_index[1]:end_index[1],
                    start_index[2]:end_index[2]] = 1
    elif image.dimension == 4:
        label_image[start_index[0]:end_index[0],
                    start_index[1]:end_index[1],
                    start_index[2]:end_index[2],
                    start_index[3]:end_index[3]] = 1

    cropped_image = ants.crop_image(image, label_image, label=1)

    return(cropped_image)
