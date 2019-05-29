
import keras.backend as K

def multilabel_dice_coefficient(y_true, y_pred):
    smoothing_factor = 1.0

    y_dims = K.int_shape(y_pred)

    number_of_labels = y_dims[len(y_dims)]

    if len(y_dims) == 3:
        # 2-D image
        y_true_permuted = K.permute_dimensions(y_true, pattern = (3, 0, 1, 2))
        y_pred_permuted = K.permute_dimensions(y_pred, pattern = (3, 0, 1, 2))
    else:
        # 3-D image
        y_true_permuted <- K.permute_dimensions(y_true, pattern = (4, 0, 1, 2, 3))
        y_pred_permuted <- K.permute_dimensions(y_pred, pattern = (4, 0, 1, 2, 3))

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

    return((2.0 * unionOverlap + smoothingFactor) /
      (1.0 + unionOverlap + smoothingFactor))

def loss_multilabel_dice_coefficient_error(y_true, y_pred):
    return -multilabel_dice_coefficient(y_true, y_pred)


def peak_signal_to_noise_ratio(y_true, y_pred):
    return(-10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0))

def loss_peak_signal_to_noise_ratio_error(y_true, y_pred):
    return(-peak_signal_to_noise_ratio(y_true, y_pred))


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

def loss_pearson_correlation_coefficient_error(y_true, y_pred)
    return(-pearson_correlation_coefficient(y_true, y_pred))
