
import tensorflow.keras.backend as K

def multilabel_dice_coefficient(y_true, y_pred, smoothing_factor=0.0):

    def multilabel_dice_coefficient_fixed(y_true, y_pred):
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

def categorical_focal_gain(y_true, y_pred, gamma=2.0, alpha=0.25):

    def categorical_focal_gain_fixed(y_true, y_pred):
        y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = y_true * K.log(y_pred)
        gain = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy
        return(K.sum(gain, axis=-1))

    return(categorical_focal_gain_fixed)

def categorical_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):

    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = y_true * K.log(y_pred)
        loss = alpha * K.pow(1.0 - y_pred, gamma) * cross_entropy
        return(-K.sum(loss, axis=-1))

    return(categorical_focal_loss_fixed)

def weighted_categorical_crossentropy(y_true, y_pred, weights):

    weights_tensor = K.variable(weights)

    def weighted_categorical_crossentropy_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights_tensor
        loss = -K.sum(loss, axis=-1)
        return(loss)

    return(weighted_categorical_crossentropy_fixed)
