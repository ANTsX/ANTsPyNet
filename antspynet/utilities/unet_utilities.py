
import numpy as np
import ants

def encode_unet(segmentations_array,
                segmentation_labels=None):

    """
    Basic one-hot transformation of segmentations array

    Arguments
    ---------
    segmentations_array : numpy array
        multi-label numpy array

    segmentation_labels : tuple or list
        Note that a background label (typically 0) needs to be included.

    Returns
    -------
    An n-d array of shape batch_size x width x height x <depth> x number_of_segmentation_labels


    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> seg = ants.kmeans_segmentation(image, 3)['segmentation']
    >>> one_hot = encode_unet(seg.numpy().astype('int'))
    """

    if segmentation_labels is None:
        segmentation_labels = np.unique(segmentations_array)

    number_of_labels = len(segmentation_labels)

    dim_segmentations = segmentations_array.shape

    image_dimension = 2
    if len(dim_segmentations) == 4:
        image_dimension = 3

    if number_of_labels < 2:
        raise ValueError("At least two segmentation labels need to be specified.")

    one_hot_array = np.zeros((*dim_segmentations, number_of_labels))
    for i in range(number_of_labels):
        per_label = np.zeros_like(segmentations_array)
        per_label[segmentations_array == segmentation_labels[i]] = 1
        if image_dimension == 2:
            one_hot_array[:,:,:,i] = per_label
        else:
            one_hot_array[:,:,:,:,i] = per_label

    return one_hot_array


def decode_unet(y_predicted,
                domain_image):

    """
    Decoding function for the u-net prediction outcome

    Arguments
    ---------
    y_predicted : an array
        Shape batch_size x width x height x <depth> x number_of_segmentation_labels

    domain_image : ANTs image
        Defines the geometry of the returned probability images

    Returns
    -------
    List of probability images.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    """

    batch_size = y_predicted.shape[0]
    number_of_labels = y_predicted.shape[-1]

    image_dimension = 2
    if len(y_predicted.shape) == 5:
        image_dimension = 3

    batch_probability_images = list()
    for i in range(batch_size):
        probability_images = list()
        for j in range(number_of_labels):
            if image_dimension == 2:
                image_array = np.squeeze(y_predicted[i,:,:,j])
            else:
                image_array = np.squeeze(y_predicted[i,:,:,:,j])
            probability_images.append(ants.from_numpy(image_array,
               origin=domain_image.origin, spacing=domain_image.spacing,
               direction=domain_image.direction))
        batch_probability_images.append(probability_images)

    return batch_probability_images

