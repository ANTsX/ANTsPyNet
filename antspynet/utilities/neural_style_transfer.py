import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import vgg19
import tensorflow.keras.backend as K

import ants

def neural_style_transfer(content_image,
                          style_images,
                          initial_combination_image=None,
                          number_of_iterations=10,
                          learning_rate=1.0,
                          total_variation_weight=8.5e-5,
                          content_weight=0.025,
                          style_image_weights=1.0,
                          content_layer_names=[
                            'block5_conv2'],
                          style_layer_names="all",
                          content_mask=None,
                          style_masks=None,
                          use_shifted_activations=True,
                          use_chained_inference=True,
                          verbose=False,
                          output_prefix=None):

    """
    The popular neural style transfer described here:

    https://arxiv.org/abs/1508.06576 and https://arxiv.org/abs/1605.04603

    and taken from François Chollet's implementation

    https://keras.io/examples/generative/neural_style_transfer/

    and titu1994's modifications:

    https://github.com/titu1994/Neural-Style-Transfer

    in order to possibly modify and experiment with medical images.

    Arguments
    ---------
    content_image : ANTsImage (1 or 3-component)
        Content (or base) image.

    style_images : ANTsImage or list of ANTsImages
        Style (or reference) image.

    initial_combination_image : ANTsImage (1 or 3-component)
        Starting point for the optimization.  Allows one to start from the
        output from a previous run.  Otherwise, start from the content image.
        Note that the original paper starts with a noise image.

    number_of_iterations : integer
        Number of gradient steps taken during optimization.

    learning_rate : float
        Parameter for Adam optimization.

    total_variation_weight : float
        A penalty on the regularization term to keep the features
        of the output image locally coherent.

    content_weight : float
        Weight of the content layers in the optimization function.

    style_image_weights : float or list of floats
        Weights of the style term in the optimization function for each
        style image.  Can either specify a single scalar to be used for
        all the images or one for each image.  The
        style term computes the sum of the L2 norm between the Gram
        matrices of the different layers (using ImageNet-trained VGG)
        of the style and content images.

    content_layer_names : list of strings
        Names of VGG layers from which to compute the content loss.

    style_layer_names : list of strings
        Names of VGG layers from which to compute the style loss.  If "all",
        the layers used are ['block1_conv1', 'block1_conv2', 'block2_conv1',
        'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
        'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3',
        'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3',
        'block5_conv4'].  This is a proposed improvement from
        https://arxiv.org/abs/1605.04603.  In the original implementation, the
        layers used are: ['block1_conv1', 'block2_conv1', 'block3_conv1',
         'block4_conv1', 'block5_conv1'].

    content_mask : ANTsImage
        Specify the region for content consideration.

    style_masks :  ANTsImage or list of ANTsImages
        Specify the region for style consideration.

    use_shifted_activations : boolean
        Use shifted activations in calculating the Gram matrix (improvement
        mentioned in https://arxiv.org/abs/1605.04603).

    use_chained_inference : boolean
        Another proposed improvement from https://arxiv.org/abs/1605.04603.

    verbose : boolean
        Print progress to the screen.

    output_prefix : string
        If specified, outputs a png image to disk at each iteration.

    Returns
    -------
    ANTs 3-component image.

    Example
    -------
    >>> image = neural_style_transfer(content_image, style_image)
    """

    def preprocess_ants_image(image, do_scale_and_center=True):
        array = None
        if image.components == 1:
            array = image.numpy()
            array = np.expand_dims(array, 2)
            array = np.repeat(array, 3, 2)
        elif image.components == 3:
            vector_image = image
            image_channels = ants.split_channels(vector_image)
            array = np.concatenate([np.expand_dims(image_channels[0].numpy(), axis=2),
                                    np.expand_dims(image_channels[1].numpy(), axis=2),
                                    np.expand_dims(image_channels[2].numpy(), axis=2)], axis=2)
        else:
            raise ValueError("Unexpected number of components.")

        if do_scale_and_center == True:
            for i in range(3):
                array[:,:,i] = (array[:,:,i] - array[:,:,i].min()) / (array[:,:,i].max() - array[:,:,i].min())
            array *= 255.0
            # RGB -> BGR
            array = array[:, :, ::-1]
            array[:, :, 0] -= 103.939
            array[:, :, 1] -= 116.779
            array[:, :, 2] -= 123.68

        array = np.expand_dims(array, 0)
        return(array)


    def postprocess_array(array, reference_image):
        array = np.squeeze(array)

        array[:, :, 0] += 103.939
        array[:, :, 1] += 116.779
        array[:, :, 2] += 123.68
        # BGR -> RGB
        array = array[:, :, ::-1]
        array = np.clip(array, 0, 255)

        image = ants.from_numpy(array,
            origin=reference_image.origin, spacing=reference_image.spacing,
            direction=reference_image.direction, has_components=True)
        return(image)


    def gram_matrix(x, shifted_activations=False):
        F = K.batch_flatten(
            K.permute_dimensions(x, (2, 0, 1)))
        if shifted_activations:
            F = F - 1
        gram = K.dot(F, K.transpose(F))
        return(gram)


    def process_mask(mask, shape):
        mask_processed = (tf.image.resize(mask, size=[shape[0], shape[1]],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)).numpy()
        mask_processed_tensor = np.empty(shape)
        for i in range(shape[2]):
            mask_processed_tensor[:, :, i] = mask_processed[:, :, 0]
        return(mask_processed_tensor)

    def style_loss(style_features, combination_features, image_shape, style_mask=None, content_mask=None):

        if content_mask is not None:
            mask_tensor = K.variable(process_mask(content_mask, combination_features.shape))
            combination_features = combination_features * K.stop_gradient(mask_tensor)
            del mask_tensor

        if style_mask is not None:
            mask_tensor = K.variable(process_mask(style_mask, style_features.shape))
            style_features = style_features * K.stop_gradient(mask_tensor)
            if content_mask is not None:
                combination_features = combination_features * K.stop_gradient(mask_tensor)
            del mask_tensor

        style_gram = gram_matrix(style_features, use_shifted_activations)
        content_gram = gram_matrix(combination_features, use_shifted_activations)
        size = image_shape[0] * image_shape[1]
        number_of_channels = 3
        loss = tf.reduce_sum(tf.square(style_gram - content_gram)) / (4.0 * (number_of_channels ** 2) * (size ** 2))
        return(loss)

    def content_loss(content_features, combination_features):
        loss = tf.reduce_sum(tf.square(content_features - combination_features))
        return(loss)

    def total_variation_loss(x):
        shape=x.shape
        a = tf.square(x[:, :(shape[1] - 1), :(shape[2] - 1), :] - x[:, 1:, :(shape[2] - 1), :])
        b = tf.square(x[:, :(shape[1] - 1), :(shape[2] - 1), :] - x[:, :(shape[1] - 1), 1:, :])
        loss = tf.reduce_sum(tf.pow(a + b, 1.25))
        return(loss)

    def compute_total_loss(content_array, style_array_list, combination_tensor,
                           feature_model, content_layer_names, style_layer_names,
                           image_shape, content_mask_tensor=None, style_mask_tensor_list=None):
        number_of_style_images = len(style_array_list)

        input_arrays = list()
        input_arrays.append(content_array)
        for i in range(number_of_style_images):
            input_arrays.append(style_array_list[i])
        input_arrays.append(combination_tensor)
        input_tensor = tf.concat(input_arrays, axis=0)

        features = feature_model(input_tensor)

        total_loss = tf.zeros(shape=())

        # content loss
        for i in range(len(content_layer_names)):
            layer_features = features[content_layer_names[i]]
            content_features = layer_features[0,:, :, :]
            combination_features = layer_features[2, :, :, :]
            total_loss = total_loss + (content_loss(content_features, combination_features) *
               content_weight / len(content_layer_names))

        # style loss
        if use_chained_inference:
            for i in range(len(style_layer_names) - 1):
                layer_features = features[style_layer_names[i]]
                style_features = layer_features[1:(number_of_style_images + 1), :, :, :]
                combination_features = layer_features[number_of_style_images + 1, :, :, :]
                loss = list()
                for j in range(number_of_style_images):
                    if style_mask_tensor_list is None:
                        loss.append(style_loss(style_features[j], combination_features, image_shape,
                                    style_mask=None, content_mask=content_mask_tensor))
                    else:
                        loss.append(style_loss(style_features[j], combination_features, image_shape,
                                    style_mask=style_mask_tensor_list[j], content_mask=content_mask_tensor))

                layer_features = features[style_layer_names[i+1]]
                style_features = layer_features[1:(number_of_style_images + 1), :, :, :]
                combination_features = layer_features[number_of_style_images + 1, :, :, :]
                loss_p1 = list()
                for j in range(number_of_style_images):
                    if style_mask_tensor_list is None:
                        loss_p1.append(style_loss(style_features[j], combination_features, image_shape,
                                       style_mask=None, content_mask=content_mask_tensor))
                    else:
                        loss_p1.append(style_loss(style_features[j], combination_features, image_shape,
                                       style_mask=style_mask_tensor_list[j], content_mask=content_mask_tensor))

                for j in range(number_of_style_images):
                    loss_difference = loss[j] - loss_p1[j]
                    total_loss = total_loss + (style_image_weights[j] * loss_difference /
                        (2 ** (len(style_layer_names) - (i + 1))))

        else:
            for i in range(len(style_layer_names)):
                layer_features = features[style_layer_names[i]]
                style_features = layer_features[1:(number_of_style_images + 1), :, :, :]
                combination_features = layer_features[number_of_style_images + 1, :, :, :]
                for j in range(number_of_style_images):
                    loss = list()
                    if style_mask_tensor_list is None:
                        loss.append(style_loss(style_features[j], combination_features, image_shape,
                                    style_mask=None, content_mask=content_mask_tensor))
                    else:
                        loss.append(style_loss(style_features[j], combination_features, image_shape,
                                    style_mask=style_mask_tensor_list[j], content_mask=content_mask_tensor))

                for j in range(number_of_style_images):
                    total_loss = total_loss + (loss[j] * style_image_weights[j] / len(style_layer_names))


        # total variation loss
        total_loss = total_loss + total_variation_weight * total_variation_loss(combination_tensor)

        return(total_loss)

    def compute_loss_and_gradients(content_array, style_array_list, combination_tensor,
      feature_model, content_layer_names, style_layer_names, image_shape,
      content_mask_tensor=None, style_mask_tensor_list=None):
        with tf.GradientTape() as tape:
            loss = compute_total_loss(content_array, style_array_list, combination_tensor,
                                      feature_model, content_layer_names,
                                      style_layer_names, image_shape, content_mask_tensor,
                                      style_mask_tensor_list)
        gradients = tape.gradient(loss, combination_tensor)
        return loss, gradients

    number_of_style_images = 1
    if isinstance(style_images, list):
        number_of_style_images = len(style_images)

    style_image_list = list()
    if number_of_style_images == 1:
        style_image_list.append(style_images)
    else:
        style_image_list = style_images

    for i in range(number_of_style_images):
        if style_image_list[i].dimension != 2:
            raise ValueError("Input style images must be 2-D.")
        if style_image_list[i].shape != content_image.shape:
            raise ValueError("Input images must have matching dimensions/shapes.")

    number_of_style_masks = 0
    style_mask_tensor_list = None
    if style_masks is not None:
        number_of_style_masks = 1
        if isinstance(style_masks, list):
            number_of_style_masks = len(style_masks)

        style_mask_tensor_list = list()
        if number_of_style_masks == 1:
            style_mask_array = (ants.threshold_image(style_masks, 0, 0, 0, 1)).numpy()
            style_mask_tensor = np.expand_dims(style_mask_array, -1)
            style_mask_tensor_list.append(style_mask_tensor)
        else:
            for i in range(len(style_masks)):
                style_mask_array = (ants.threshold_image(style_masks[i], 0, 0, 0, 1)).numpy()
                style_mask_tensor = np.expand_dims(style_mask_array, -1)
                style_mask_tensor_list.append(style_mask_tensor)

    if number_of_style_masks > 0 and number_of_style_images != number_of_style_masks:
        raise ValueError("The number of style images/masks are not the same.")

    if isinstance(style_image_weights, (int, float)):
        style_image_weights = [style_image_weights] * len(style_image_list)
    else:
        if len(style_image_weights) == 1:
            style_image_weights = style_image_weights * len(style_image_list)
        elif not len(style_image_weights) == len(style_image_list):
            raise ValueError("Length of style weights must be 1 or the number of style images.")

    if content_image.dimension != 2:
        raise ValueError("Input content image must be 2-D.")

    content_mask_tensor = None
    if content_mask is not None:
        content_mask_array = (ants.threshold_image(content_mask, 0, 0, 0, 1)).numpy()
        content_mask_tensor = np.expand_dims(content_mask_array, -1)

    if style_layer_names == "all":
        style_layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1',
            'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3',
            'block3_conv4', 'block4_conv1', 'block4_conv2', 'block4_conv3',
            'block4_conv4', 'block5_conv1', 'block5_conv2', 'block5_conv3',
            'block5_conv4']

    model = vgg19.VGG19(weights="imagenet", include_top=False)

    outputs_dictionary = dict([(layer.name, layer.output) for layer in model.layers])
    # shapes_dictionary = dict([(layer.name, layer.output_shape) for layer in model.layers])

    feature_model = tf.keras.Model(inputs=model.inputs, outputs=outputs_dictionary)

    # Preprocess data
    content_array = preprocess_ants_image(content_image)
    style_array_list = list()
    for i in range(number_of_style_images):
        style_array_list.append(preprocess_ants_image(style_image_list[i]))

    image_shape = (content_array.shape[1], content_array.shape[2], 3)

    combination_tensor = None
    if initial_combination_image is None:
        combination_tensor = tf.Variable(np.copy(content_array))
    else:
        initial_combination_tensor = preprocess_ants_image(initial_combination_image, do_scale_and_center=False)
        combination_tensor = tf.Variable(initial_combination_tensor)

    if not image_shape == (combination_tensor.shape[1], combination_tensor.shape[2], 3):
        raise ValueError("Initial combination image size does not match content image.")

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=0.1)

    for i in range(number_of_iterations):
        start_time = time.time()
        loss, gradients = compute_loss_and_gradients(content_array, style_array_list,
                              combination_tensor, feature_model, content_layer_names,
                              style_layer_names, image_shape, content_mask_tensor,
                              style_mask_tensor_list)
        end_time = time.time()
        if verbose == True:
            print("Iteration %d of %d: total loss = %.2f (elapsed time = %ds)" %
              (i, number_of_iterations, loss, end_time - start_time))
        optimizer.apply_gradients([(gradients, combination_tensor)])

        if not output_prefix == None:
            combination_array = combination_tensor.numpy()
            combination_image = postprocess_array(combination_array, content_image)
            combination_rgb = combination_image.vector_to_rgb()
            ants.image_write(combination_rgb, output_prefix + "_iteration%d.png" % (i + 1))

    combination_array = combination_tensor.numpy()
    combination_image = postprocess_array(combination_array, content_image)
    return(combination_image)
