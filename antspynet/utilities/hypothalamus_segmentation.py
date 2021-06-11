
import numpy as np
import ants

def hypothalamus_segmentation(t1,
                              antsxnet_cache_directory=None,
                              verbose=False):

    """
    Hypothalamus and subunits segmentation

    Described here:

        https://pubmed.ncbi.nlm.nih.gov/32853816/

    ported from the original implementation

        https://github.com/BBillot/hypothalamus_seg

    Subunits labeling:

    Label 1:  left anterior-inferior
    Label 2:  left anterior-superior
    Label 3:  left posterior
    Label 4:  left tubular inferior
    Label 5:  left tubular superior
    Label 6:  right anterior-inferior
    Label 7:  right anterior-superior
    Label 8:  right posterior
    Label 9:  right tubular inferior
    Label 10: right tubular superior

    Arguments
    ---------
    t1 : ANTsImage
        input 3-D T1 brain image.

    antsxnet_cache_directory : string
        Destination directory for storing the downloaded template and model weights.
        Since these can be resused, if is None, these data will be downloaded to a
        ~/.keras/ANTsXNet/.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    Hypothalamus segmentation (and subunits) probability images

    Example
    -------
    >>> image = ants.image_read("t1.nii.gz")
    >>> hypo = hypothalamus_segmentation(image)
    """

    from ..architectures import create_hypothalamus_unet_model_3d
    from ..utilities import get_pretrained_network

    if t1.dimension != 3:
        raise ValueError( "Image dimension must be 3." )

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    classes = ("background",
               "left anterior-inferior",
               "left anterior-superior",
               "left posterior",
               "left tubular inferior",
               "left tubular superior",
               "right anterior-inferior",
               "right anterior-superior",
               "right posterior",
               "right tubular inferior",
               "right tubular superior")

    ################################
    #
    # Rotate to proper orientation
    #
    ################################

    reference_image = ants.make_image((256, 256, 256),
                                      voxval=0,
                                      spacing=(1, 1, 1),
                                      origin=(0, 0, 0),
                                      direction=np.diag((-1.0, -1.0, 1.0)))
    center_of_mass_reference = ants.get_center_of_mass(reference_image + 1)
    center_of_mass_image = ants.get_center_of_mass(t1 * 0 + 1)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_reference)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_reference), translation=translation)
    xfrm_inv = xfrm.invert()

    crop_image = ants.image_clone(t1) * 0 + 1
    crop_image = ants.apply_ants_transform_to_image(xfrm, crop_image, reference_image)
    crop_image = ants.crop_image(crop_image, label_image=crop_image, label=1)

    t1_warped = ants.apply_ants_transform_to_image(xfrm, t1, crop_image)

    ################################
    #
    # Normalize intensity
    #
    ################################

    t1_warped = ((t1_warped - t1_warped.min()) / (t1_warped.max() - t1_warped.min()))

    ################################
    #
    # Build models and load weights
    #
    ################################

    if verbose == True:
        print("Hypothalamus:  retrieving model weights.")

    unet_model = create_hypothalamus_unet_model_3d(t1_warped.shape)

    weights_file_name = get_pretrained_network("hypothalamus", antsxnet_cache_directory=antsxnet_cache_directory)
    unet_model.load_weights(weights_file_name)

    ################################
    #
    # Do prediction
    #
    ################################

    if verbose == True:
        print("Prediction.")

    batchX = np.zeros((1, *t1_warped.shape, 1))
    batchX[0,:,:,:,0] = t1_warped.numpy()

    predicted_data = unet_model.predict(batchX, verbose=verbose)

    probability_images = list()
    for i in range(len(classes)):
        if verbose == True:
            print("Processing image", classes[i])

        probability_image = ants.from_numpy(np.squeeze(predicted_data[0,:,:,:,i]),
            spacing=t1_warped.spacing, origin=t1_warped.origin,
            direction=t1_warped.direction)
        probability_images.append(xfrm_inv.apply_to_image(probability_image, t1))

    image_matrix = ants.image_list_to_matrix(probability_images, t1 * 0 + 1)
    segmentation_matrix = np.argmax(image_matrix, axis=0)
    segmentation_image = ants.matrix_to_images(
        np.expand_dims(segmentation_matrix, axis=0), t1 * 0 + 1)[0]

    return_dict = {'segmentation_image' : segmentation_image,
                   'probability_images' : probability_images}
    return(return_dict)

