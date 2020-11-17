import numpy as np
import tensorflow as tf
import ants

def tid_neural_image_assessment(image,
                                mask=None,
                                patch_size=101,
                                stride_length=None,
                                padding_size=0,
                                dimensions_to_predict=0,
                                antsxnet_cache_directory=None,
                                which_model="tidsQualityAssessment",
                                verbose=False):

    """
    Perform MOS-based assessment of an image.

    Use a ResNet architecture to estimate image quality in 2D or 3D using subjective
    QC image databases described in

    https://www.sciencedirect.com/science/article/pii/S0923596514001490

    or

    https://doi.org/10.1109/TIP.2020.2967829

    where the image assessment is either "global", i.e., a single number or an image
    based on the specified patch size.  In the 3-D case, neighboring slices are used
    for each estimate.  Note that parameters should be kept as consistent as possible
    in order to enable comparison.  Patch size should be roughly 1/12th to 1/4th of
    image size to enable locality. A global estimate can be gained by setting
    patch_size = "global".

    Arguments
    ---------
    image : ANTsImage (2-D or 3-D)
        input image.

    mask : ANTsImage (2-D or 3-D)
        optional mask for designating calculation ROI.

    patch_size : integer
        prime number of patch_size.  101 is good.  Otherwise, choose "global" for a single
        global estimate of quality.

    stride_length : integer or vector of image dimension length
        optional value to speed up computation (typically less than patch size).

    padding_size : positive or negative integer or vector of image dimension length
        de(padding) to remove edge effects.

    dimensions_to_predict : integer or vector
        if image dimension is 3, this parameter specifies which dimensions should be used for
        prediction.  If more than one dimension is specified, the results are averaged.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to
        ~/.keras/ANTsXNet/.

    which_model : string
        model type e.g. string tidsQualityAssessment, koniqMS, koniqMS2 or koniqMS3 where
        the former predicts mean opinion score (MOS) and MOS standard deviation and
        the latter koniq models predict mean opinion score (MOS) and sharpness.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    List of QC results predicting both both human rater's mean and standard
    deviation of the MOS ("mean opinion scores") or sharpness depending on the
    selected network.  Both aggregate and spatial scores are returned, the latter
    in the form of an image.

    Example
    -------
    >>> image = ants.image_read(ants.get_data("r16"))
    >>> mask = ants.get_mask(image)
    >>> tid = tid_neural_image_assessment(image, mask=mask, patch_size=101, stride_length=7)
    """

    from ..utilities import get_pretrained_network
    from ..utilities import pad_or_crop_image_to_size
    from ..utilities import extract_image_patches
    from ..utilities import reconstruct_image_from_patches

    def is_prime(n):
        if n == 2 or n == 3:
            return True
        if n < 2 or n % 2 == 0:
            return False
        if n < 9:
            return True
        if n%3 == 0:
            return False
        r = int(n ** 0.5)
        f = 5
        while f <= r:
            if n % f == 0:
                return False
            if n % (f + 2) == 0:
                return False
            f += 6
        return True

    valid_models = ("tidsQualityAssessment", "koniqMS", "koniqMS2", "koniqMS3")
    if not which_model in valid_models:
        raise ValueError("Please pass valid model")

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    if verbose == True:
        print("Neural QA:  retreiving model and weights.")

    is_koniq = "koniq" in which_model
    model_and_weights_file_name = get_pretrained_network(which_model, antsxnet_cache_directory=antsxnet_cache_directory)
    tid_model = tf.keras.models.load_model(model_and_weights_file_name, compile=False)

    padding_size_vector = padding_size
    if isinstance(padding_size, int):
        padding_size_vector = np.repeat(padding_size, image.dimension)
    elif len(padding_size) == 1:
        padding_size_vector = np.repeat(padding_size[0], image.dimension)

    if isinstance(dimensions_to_predict, int):
        dimensions_to_predict = (dimensions_to_predict,)

    padded_image_size = image.shape + padding_size_vector
    padded_image = pad_or_crop_image_to_size(image, padded_image_size)

    number_of_channels = 3

    if stride_length is None and patch_size != "global":
        stride_length = round(min(patch_size) / 2)
        if image.dimension == 3:
            stride_length = (stride_length, stride_length, 1)

    ###############
    #
    #  Global
    #
    ###############

    if patch_size == 'global':
        if which_model == "tidsQualityAssessment":
            evaluation_image = ants.iMath(padded_image, "Normalize") * 255

        if is_koniq:
            evaluation_image = ants.iMath(padded_image, "Normalize") * 2.0 - 1.0

        if image.dimension == 2:
            batchX = np.zeros((1, evaluation_image.shape, number_of_channels))
            for k in range(3):
                batchX[0,:,:,k] = evaluation_image.numpy()
            predicted_data = tid_model.predict(batchX, verbose=verbose)

            if which_model == "tidsQualityAssessment":
                return_dict = {'MOS' : None,
                               'MOS.standardDeviation' : None,
                               'MOS.mean' : predicted_data[0, 0],
                               'MOS.standardDeviationMean' : predicted_data[0, 1]
                              }
                return(return_dict)

            elif is_koniq:
                return_dict = {'MOS.mean' : predicted_data[0, 0],
                               'sharpness.mean' : predicted_data[0, 1]
                              }
                return(return_dict)

        elif image.dimension == 3:
            mos_mean = 0
            mos_standard_deviation = 0

            x = tuple(range(image.dimension))
            for d in range(len(dimensions_to_predict)):
                not_padded_image_size = list(padded_image_size)
                del(not_padded_image_size[dimensions_to_predict[d]])
                batchX = np.zeros((padded_image_size[dimensions_to_predict[d]],
                    (*tuple(not_padded_image_size)), number_of_channels))

                batchX[0,:,:,0] = (ants.slice_image(evaluation_image, axis=0, idx=dimensions_to_predict[d])).numpy()
                batchX[0,:,:,1] = (ants.slice_image(evaluation_image, axis=0, idx=dimensions_to_predict[d])).numpy()
                batchX[0,:,:,2] = (ants.slice_image(evaluation_image, axis=1, idx=dimensions_to_predict[d])).numpy()
                for i in range(1, padded_image_size[dimensions_to_predict[d]] - 1):
                    batchX[i,:,:,0] = (ants.slice_image(evaluation_image, axis=i-1, idx=dimensions_to_predict[d])).numpy()
                    batchX[i,:,:,1] = (ants.slice_image(evaluation_image, axis=i  , idx=dimensions_to_predict[d])).numpy()
                    batchX[i,:,:,2] = (ants.slice_image(evaluation_image, axis=i+1, idx=dimensions_to_predict[d])).numpy()
                batchX[padded_image_size[dimensions_to_predict[d]],:,:,0] = (
                    ants.slice_image(evaluation_image, axis=padded_image_size[dimensions_to_predict[d]]-1,
                        idx=dimensions_to_predict[d])).numpy()
                batchX[padded_image_size[dimensions_to_predict[d]],:,:,1] = (
                    ants.slice_image(evaluation_image, axis=padded_image_size[dimensions_to_predict[d]],
                        idx=dimensions_to_predict[d])).numpy()
                batchX[padded_image_size[dimensions_to_predict[d]],:,:,2] = (
                    ants.slice_image(evaluation_image, axis=padded_image_size[dimensions_to_predict[d]],
                        idx=dimensions_to_predict[d])).numpy()

                predicted_data = tid_model.predict(batchX, verbose=verbose)
                mos_mean += predicted_data[0, 0]
                mos_standard_deviation += predicted_data[0, 1]

            mos_mean /= len(dimensions_to_predict)
            mos_standard_deviation /= len(dimensions_to_predict)
            if which_model == "tidsQualityAssessment":
                return_dict = {'MOS.mean' : mos_mean,
                               'MOS.standardDeviationMean' : mos_standard_deviation
                              }
                return(return_dict)
            elif is_koniq:
                return_dict = {'MOS.mean' : mos_mean,
                               'sharpness.mean' : mos_standard_deviation
                              }
                return(return_dict)

    ###############
    #
    #  Patchwise
    #
    ###############

    else:

        evaluation_image = padded_image

        if not is_prime(patch_size):
            print("patch_size should be a prime number:  13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97...")

        stride_length_vector = stride_length
        if isinstance(stride_length, int):
            if image.dimension == 2:
                stride_length_vector = (stride_length, stride_length)
        elif len(stride_length) == 1:
            if image.dimension == 2:
                stride_length_vector = (stride_length[0], stride_length[0])

        patch_size_vector = (patch_size, patch_size)

        if image.dimension == 2:
            dimensions_to_predict = (1,)

        permutations = list()

        mos = image * 0
        mos_standard_deviation = image * 0

        for d in range(len(dimensions_to_predict)):
            if image.dimension == 3:
                permutations.append((0, 1, 2))
                permutations.append((0, 2, 1))
                permutations.append((1, 2, 0))

                if dimensions_to_predict[d] == 0:
                    patch_size_vector = (patch_size, patch_size, number_of_channels)
                    if isinstance(stride_length, int):
                        stride_length_vector = (stride_length, stride_length, 1)
                elif dimensions_to_predict[d] == 1:
                    patch_size_vector = (patch_size, number_of_channels, patch_size)
                    if isinstance(stride_length, int):
                        stride_length_vector = (stride_length, 1, stride_length)
                elif dimensions_to_predict[d] == 2:
                    patch_size_vector = (number_of_channels, patch_size, patch_size)
                    if isinstance(stride_length, int):
                        stride_length_vector = (1, stride_length, stride_length)
                else:
                    raise ValueError("dimensions_to_predict elements should be 1, 2, and/or 3 for 3-D image.")

            patches = extract_image_patches(evaluation_image, patch_size=patch_size_vector,
            stride_length=stride_length_vector, return_as_array=False)

            batchX = np.zeros((len(patches), patch_size, patch_size, number_of_channels))

            is_good_patch = np.repeat(False, len(patches))
            for i in range(len(patches)):
                if patches[i].var() > 0:
                    is_good_patch[i] = True
                    patch_image = patches[i]
                    patch_image = patch_image - patch_image.min()

                    if patch_image.max() > 0:
                        if which_model == "tidsQualityAssessment":
                            patch_image = patch_image / patch_image.max() * 255
                        elif is_koniq:
                            patch_image = patch_image / patch_image.max() * 2.0 - 1.0

                    if image.dimension == 2:
                        for j in range(number_of_channels):
                            batchX[i,:,:,j] = patch_image
                    elif image.dimension == 3:
                        batchX[i,:,:,:] = np.transpose(np.squeeze(patch_image.numpy()), permutations[dimensions_to_predict[d]])

            good_batchX = batchX[is_good_patch,:,:,:]
            predicted_data = tid_model.predict(good_batchX, verbose=verbose)

            patches_mos = list()
            patches_mos_standard_deviation = list()

            zero_patch_image = patch_image * 0

            count = 0
            for i in range(len(patches)):
                if is_good_patch[i]:
                    patches_mos.append(zero_patch_image + predicted_data[count, 0])
                    patches_mos_standard_deviation.append(zero_patch_image + predicted_data[count, 1])
                    count += 1
                else:
                    patches_mos.append(zero_patch_image)
                    patches_mos_standard_deviation.append(zero_patch_image)

            mos += pad_or_crop_image_to_size(reconstruct_image_from_patches(
                patches_mos, evaluation_image, stride_length=stride_length_vector), image.shape)
            mos_standard_deviation += pad_or_crop_image_to_size(reconstruct_image_from_patches(
                patches_mos_standard_deviation, evaluation_image, stride_length=stride_length_vector), image.shape)

        mos /= len(dimensions_to_predict)
        mos_standard_deviation /= len(dimensions_to_predict)

        if mask is None:

            if which_model == "tidsQualityAssessment":
                return_dict = {'MOS' : mos,
                               'MOS.standardDeviation' : mos_standard_deviation,
                               'MOS.mean' : mos.mean(),
                               'MOS.standardDeviationMean' : mos_standard_deviation.mean()
                              }
                return(return_dict)

            elif is_koniq:
                return_dict = {'MOS' : mos,
                               'sharpness' : mos_standard_deviation,
                               'MOS.mean' : mos.mean(),
                               'sharpness.mean' : mos_standard_deviation.mean()
                              }
                return(return_dict)

        else:

            if which_model == "tidsQualityAssessment":
                return_dict = {'MOS' : mos * mask,
                               'MOS.standardDeviation' : mos_standard_deviation * mask,
                               'MOS.mean' : (mos[mask >= 0.5]).mean(),
                               'MOS.standardDeviationMean' : (mos_standard_deviation[mask >= 0.5]).mean()
                              }
                return(return_dict)

            elif is_koniq:
                return_dict = {'MOS' : mos * mask,
                               'sharpness' : mos_standard_deviation * mask,
                               'MOS.mean' : (mos[mask >= 0.5]).mean(),
                               'sharpness.mean' : (mos_standard_deviation[mask >= 0.5]).mean()
                              }
                return(return_dict)













