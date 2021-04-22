import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import scipy as sp

def binary_dice_coefficient(smoothing_factor=0.0):

    """
    Binary dice segmentation loss.

    Note:  Assumption is that y_true is *not* a one-hot representation
    of the segmentation batch.  For use with e.g., sigmoid activation.

    Arguments
    ---------

    smoothing_factor : float
        Used to smooth value during optimization

    Returns
    -------
    Loss value (negative Dice coefficient).

    """

    def binary_dice_coefficient_fixed(y_true, y_pred):

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return(-1.0 * (2.0 * intersection + smoothing_factor)/
           (K.sum(y_true_f) + K.sum(y_pred_f) + smoothing_factor))

    return(binary_dice_coefficient_fixed)

def multilabel_dice_coefficient(dimensionality=3, smoothing_factor=0.0):

    """
    Multi-label dice segmentation loss

    Note:  Assumption is that y_true is a one-hot representation
    of the segmentation batch.  The background (label 0) should
    be included but is not used in the calculation.  For use with
    e.g., softmax activation.

    Arguments
    ---------
    dimensionality : dimensionality
        Image dimension

    smoothing_factor : float
        Used to smooth value during optimization

    Returns
    -------
    Loss value (negative Dice coefficient).

    Example
    -------
    >>> import ants
    >>> import antspynet
    >>> import tensorflow as tf
    >>> import numpy as np
    >>>
    >>> r16 = ants.image_read(ants.get_ants_data("r16"))
    >>> r16_seg = ants.kmeans_segmentation(r16, 3)['segmentation']
    >>> r16_array = np.expand_dims(r16_seg.numpy(), axis=0)
    >>> r16_tensor = tf.convert_to_tensor(antspynet.encode_unet(r16_array, (0, 1, 2, 3)))
    >>>
    >>> r64 = ants.image_read(ants.get_ants_data("r64"))
    >>> r64_seg = ants.kmeans_segmentation(r64, 3)['segmentation']
    >>> r64_array = np.expand_dims(r64_seg.numpy(), axis=0)
    >>> r64_tensor = tf.convert_to_tensor(antspynet.encode_unet(r64_array, (0, 1, 2, 3)))
    >>>
    >>> dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=2)
    >>> loss_value = dice_loss(r16_tensor, r64_tensor).numpy()
    >>> # Compare with...
    >>> ants.label_overlap_measures(r16_seg, r64_seg)
    """

    def multilabel_dice_coefficient_fixed(y_true, y_pred):
        y_dims = K.int_shape(y_pred)

        number_of_labels = y_dims[len(y_dims)-1]

        if dimensionality == 2:
            # 2-D image
            y_true_permuted = K.permute_dimensions(y_true, pattern = (3, 0, 1, 2))
            y_pred_permuted = K.permute_dimensions(y_pred, pattern = (3, 0, 1, 2))
        elif dimensionality == 3:
            # 3-D image
            y_true_permuted = K.permute_dimensions(y_true, pattern = (4, 0, 1, 2, 3))
            y_pred_permuted = K.permute_dimensions(y_pred, pattern = (4, 0, 1, 2, 3))
        else:
            raise ValueError("Specified dimensionality not implemented.")

        y_true_label = K.gather(y_true_permuted, indices = (1))
        y_pred_label = K.gather(y_pred_permuted, indices = (1))

        y_true_label_f = K.flatten(y_true_label)
        y_pred_label_f = K.flatten(y_pred_label)
        intersection = y_true_label_f * y_pred_label_f
        union = y_true_label_f + y_pred_label_f - intersection

        numerator = K.sum(intersection)
        denominator = K.sum(union)

        if number_of_labels > 2:
            for j in range(2, number_of_labels):
                y_true_label = K.gather(y_true_permuted, indices = (j))
                y_pred_label = K.gather(y_pred_permuted, indices = (j))
                y_true_label_f = K.flatten(y_true_label)
                y_pred_label_f = K.flatten(y_pred_label)

                intersection = y_true_label_f * y_pred_label_f
                union = y_true_label_f + y_pred_label_f - intersection

                numerator = numerator + K.sum(intersection)
                denominator = denominator + K.sum(union)

        unionOverlap = numerator / denominator

        return(-1.0 * (2.0 * unionOverlap + smoothing_factor) /
        (1.0 + unionOverlap + smoothing_factor))

    return(multilabel_dice_coefficient_fixed)

def peak_signal_to_noise_ratio(y_true, y_pred):
    return(-10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0))

def pearson_correlation_coefficient(y_true, y_pred):
    N = K.sum(K.ones_like(y_true))

    sum_x = K.sum(y_true)
    sum_y = K.sum(y_pred)
    sum_x_squared = K.sum(K.square(y_true))
    sum_y_squared = K.sum(K.square(y_pred))
    sum_xy = K.sum(y_true * y_pred)

    numerator = sum_xy - (sum_x * sum_y / N)
    denominator = K.sqrt((sum_x_squared - K.square(sum_x) / N) *
      (sum_y_squared - K.square(sum_y) / N))

    coefficient = numerator / denominator

    return(coefficient)

def categorical_focal_loss(gamma=2.0, alpha=0.25):

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = y_true * K.log(y_pred)
        loss = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy
        return(-K.sum(loss, axis=-1))

    return(categorical_focal_loss_fixed)

def weighted_categorical_crossentropy(weights):

    weights_tensor = K.variable(weights)

    def weighted_categorical_crossentropy_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights_tensor
        loss = -K.sum(loss, axis=-1)
        return(loss)

    return(weighted_categorical_crossentropy_fixed)

def multilabel_surface_loss(dimensionality=3):

    def multilabel_surface_loss_fixed(y_true, y_pred):
        def calculate_residual_distance_map(segmentation):
            residual_distance = np.zeros_like(segmentation)

            positive_mask = segmentation.astype(np.bool)
            if positive_mask.any():
                negative_mask = ~positive_mask
                residual_distance = \
                    (sp.ndimage.distance_transform_edt(negative_mask) * negative_mask -
                    (sp.ndimage.distance_transform_edt(positive_mask) - 1) * positive_mask)

            return(residual_distance)

        def calculate_batchwise_residual_distance_maps(y_true):
            y_true_numpy = y_true.numpy()
            return(np.array([calculate_residual_distance_map(y)
                for y in y_true_numpy]).astype(np.float32))

        y_true_distance_map = tf.py_function(
            func=calculate_batchwise_residual_distance_maps,
            inp=[y_true],
            Tout=tf.float32)

        product = y_pred * y_true_distance_map
        return(K.mean(product))

    return(multilabel_surface_loss_fixed)


def maximum_mean_discrepancy(sigma=1.0):

    def maximum_mean_discrepancy_fixed(y_true, y_pred):

        x = y_true
        y = y_pred

        def compute_kernel(x, y, sigma=1.0):

            x_size = K.shape(x)[0]
            y_size = K.shape(x)[1]
            dim = K.shape(x)[1]
            x_tiled = K.tile(K.reshape(x, K.stack([x_size, 1, dim])), K.stack([1, y_size, 1]))
            y_tiled = K.tile(K.reshape(y, K.stack([1, y_size, dim])), K.stack([x_size, 1, 1]))

            denominator = 2.0 * K.square(sigma)
            kernel_value = K.exp(-K.mean(K.square(x_tiled - y_tiled) / denominator, axis=3) / K.cast( dim, 'float32'))
            return kernel_value

        x_kernel = compute_kernel(x, x, sigma)
        y_kernel = compute_kernel(y, y, sigma)
        xy_kernel = compute_kernel(x, y, sigma)

        mmd_value = K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
        return mmd_value

    return(maximum_mean_discrepancy_fixed)
