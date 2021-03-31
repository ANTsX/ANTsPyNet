import ants
import numpy as np
import tensorflow.keras.backend as K

from . import extract_image_patches
from . import regression_match_image

from tensorflow.keras.models import load_model
import tensorflow as tf

import time
from os import path


def mse(x, y=None):
    """
    Mean square error of a single image or between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value.

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = mse(r16, r64)
    """

    if y is None:
        x2 = x ** 2
        return x2.mean()
    else:
        diff2 = (x - y) ** 2
        return diff2.mean()


def mae(x, y=None):
    """
    Mean absolute error of a single image or between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = mae(r16, r64)
    """

    if y is None:
        xabs = x.abs()
        return xabs.mean()
    else:
        diffabs = (x-y).abs().mean()
        return diffabs.mean()

def psnr(x, y):
    """
    Peak signal-to-noise ratio between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = psnr(r16, r64)
    """

    value = 20 * np.log10(x.max()) - 10 * np.log10(mse(x, y))
    return value


def ssim(x, y, K=(0.01, 0.03)):
    """
    Structural similarity index (SSI) between two images.

    Implementation of the SSI quantity for two images proposed in

    Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. "Image quality
    assessment: from error visibility to structural similarity". IEEE TIP.
    13 (4): 600–612.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    K : tuple of length 2
        tuple which contain SSI parameters meant to stabilize the formula
        in case of weak denominators.

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = psnr(r16, r64)
    """

    global_max = np.max( ( x.max(), y.max()) )
    global_min = np.abs(min( ( x.min(), y.min())) )
    L = global_max - global_min

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    C3 = C2 / 2

    mu_x = x.mean()
    mu_y = y.mean()

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = (x * x).mean() - mu_x_sq
    sigma_y_sq = (y * y).mean() - mu_y_sq
    sigma_xy = (x * y).mean() - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    SSI = numerator / denominator

    return SSI


def gmsd(x, y):
    """
    Gradient magnitude similarity deviation

    A fast and simple metric that correlates to perceptual quality.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = gmsd(r16, r64)
    """

    gx = ants.iMath(x, "Grad")
    gy = ants.iMath(y, "Grad")

    # see eqn 4 - 6 in https://arxiv.org/pdf/1308.3052.pdf

    constant = 0.0026
    gmsd_numerator = gx * gy * 2.0 + constant
    gmsd_denominator = gx ** 2 + gy ** 2 + constant
    gmsd = gmsd_numerator / gmsd_denominator

    product_dimension = 1
    for i in range(len(x.shape)):
        product_dimension *= x.shape[i]
    prefactor = 1.0 / product_dimension

    return np.sqrt(prefactor * ((gmsd - gmsd.mean()) ** 2).sum())


def apply_super_resolution_model_to_image(
    image,
    model,
    target_range=(-127.5, 127.5),
    batch_size=32,
    regression_order=None,
    verbose=False,
):

    """
    Apply a pretrained deep back projection model for super resolution.
    Helper function for applying a pretrained deep back projection model.
    Apply a patch-wise trained network to perform super-resolution. Can be applied
    to variable sized inputs. Warning: This function may be better used on CPU
    unless the GPU can accommodate the full image size. Warning 2: The global
    intensity range (min to max) of the output will match the input where the
    range is taken over all channels.

    Arguments
    ---------
    image : ANTs image
        input image.

    model : keras object or string
        pretrained keras model or filename.

    target_range : 2-element tuple
        a tuple or array defining the (min, max) of the input image
        (e.g., -127.5, 127.5).  Output images will be scaled back to original
        intensity. This range should match the mapping used in the training
        of the network.

    batch_size : integer
        Batch size used for the prediction call.

    regression_order : integer
        If specified, Apply the function regression_match_image with
        poly_order=regression_order.

    verbose : boolean
        If True, show status messages.

    Returns
    -------
    Super-resolution image upscaled to resolution specified by the network.

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data('r16'))
    >>> image_sr = apply_super_resolution_model_to_image(image, get_pretrained_network("dbpn4x"))
    """
    tflite_flag = False
    channel_axis = 0
    if K.image_data_format() == "channels_last":
        channel_axis = -1

    if target_range[0] > target_range[1]:
        target_range = target_range[::-1]

    start_time = time.time()
    if isinstance(model, str):
        if path.isfile(model):
            if verbose:
                print("Load model.")
            if path.splitext(model)[1] == '.tflite':
                interpreter = tf.lite.Interpreter(model)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                shape_length = len(interpreter.get_input_details()[0]['shape'])
                tflite_flag = True
            else:    
                model = load_model(model)
                shape_length = len(model.input_shape)

            if verbose:
                elapsed_time = time.time() - start_time
                print("  (elapsed time: ", elapsed_time, ")")
        else:
            raise ValueError("Model not found.")
    else:
        shape_length = len(model.input_shape)

    
    if shape_length < 4 | shape_length > 5:
        raise ValueError("Unexpected input shape.")
    else:
        if shape_length == 5 & image.dimension != 3:
            raise ValueError("Expecting 3D input for this model.")
        elif shape_length == 4 & image.dimension != 2:
            raise ValueError("Expecting 2D input for this model.")

    if channel_axis == -1:
        channel_axis < shape_length
    if  tflite_flag:
        channel_size = interpreter.get_input_details()[0]['shape'][channel_axis]
    else:
        channel_size = model.input_shape[channel_axis]

    if channel_size != image.components:
        raise ValueError(
            "Channel size of model",
            str(channel_size),
            "does not match ncomponents=",
            str(image.components),
            "of the input image.",
        )

    image_patches = extract_image_patches(
        image,
        patch_size=image.shape,
        max_number_of_patches=1,
        stride_length=image.shape,
        return_as_array=True,
    )
    if image.components == 1:
        image_patches = np.expand_dims(image_patches, axis=-1)

    image_patches = image_patches - image_patches.min()
    image_patches = (
        image_patches / image_patches.max() * (target_range[1] - target_range[0])
        + target_range[0]
    )

    if verbose:
        print("Prediction")

    start_time = time.time()

    if  tflite_flag:
        image_patches = image_patches.astype('float32')
        interpreter.set_tensor(input_details[0]['index'], image_patches)
        interpreter.invoke()
        out = interpreter.tensor(output_details[0]['index'])
        prediction = out()
    else:
        prediction = model.predict(image_patches, batch_size=batch_size)

    if verbose:
        elapsed_time = time.time() - start_time
        print("  (elapsed time: ", elapsed_time, ")")

    if verbose:
        print("Reconstruct intensities")

    intensity_range = image.range()
    prediction = prediction - prediction.min()
    prediction = (
        prediction / prediction.max() * (intensity_range[1] - intensity_range[0])
        + intensity_range[0]
    )

    def slice_array_channel(input_array, slice, channel_axis=-1):
        if channel_axis == 0:
            if shape_length == 4:
                return input_array[slice, :, :, :]
            else:
                return input_array[slice, :, :, :, :]
        else:
            if shape_length == 4:
                return input_array[:, :, :, slice]
            else:
                return input_array[:, :, :, :, slice]

    expansion_factor = np.asarray(prediction.shape) / np.asarray(image_patches.shape)
    if channel_axis == 0:
        FIXME

    expansion_factor = expansion_factor[1 : (len(expansion_factor) - 1)]

    if verbose:
        print("ExpansionFactor:", str(expansion_factor))

    if image.components == 1:
        image_array = slice_array_channel(prediction, 0, channel_axis)
        prediction_image = ants.make_image(
            (np.asarray(image.shape) * np.asarray(expansion_factor)).astype(int),
            image_array,
        )
        if regression_order is not None:
            reference_image = ants.resample_image_to_target(image, prediction_image)
            prediction_image = regression_match_image(
                prediction_image, reference_image, poly_order=regression_order
            )
    else:
        image_component_list = list()
        for k in range(image.components):
            image_array = slice_array_channel(prediction, k, channel_axis)
            image_component_list.append(
                ants.make_image(
                    (np.asarray(image.shape) * np.asarray(expansion_factor)).astype(
                        int
                    ),
                    image_array,
                )
            )
        prediction_image = ants.merge_channels(image_component_list)

    prediction_image = ants.copy_image_info(image, prediction_image)
    ants.set_spacing(
        prediction_image,
        tuple(np.asarray(image.spacing) / np.asarray(expansion_factor)),
    )

    return prediction_image
