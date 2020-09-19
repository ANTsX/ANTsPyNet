import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import vgg19

import ants

def neural_style_transfer(content_image,
                          style_image,
                          initial_combination_image=None,
                          number_of_iterations=4000,
                          initial_learning_rate=10.0,
                          total_variation_weight=1e-6,
                          content_weight=2.5e-8,
                          style_weight=1e-6,
                          style_layer_names=[
                                "block1_conv1",
                                "block2_conv1",
                                "block3_conv1",
                                "block4_conv1",
                                "block5_conv1",
                            ],
                          content_layer_names=[
                              "block5_conv2"
                            ],
                          verbose=False):

    """
    The popular neural style transfer described here:

    https://arxiv.org/abs/1508.06576

    and taken from François Chollet's implementation

    https://keras.io/examples/generative/neural_style_transfer/

    in order to possibly modify and experiment with medical images.

    Arguments
    ---------
    content_image : ANTsImage (1 or 3-component)
        Content (or base) image.

    style_image : ANTsImage
        Style (or reference) image.

    initial_combination_image : ANTsImage (1 or 3-component)
        Starting point for the optimization.  Allows one to start from the
        output from a previous run.  Otherwise, start from the content image.

    number_of_iterations : integer
        Number of gradient steps taken during optimization.

    initial_iteration_rate : float
        Parameter for SGD optimization with exponential decay.

    total_variation_weight : float
        A penalty on the regularization term to keep the features
        of the output image locally coherent.

    style_weight : float
        Weight of the style term in the optimization function.  The
        style term computes the sum of the L2 norm between the Gram
        matrices of the different layers (using ImageNet-trained VGG)
        of the style and content images.

    style_layer_names : list of strings
        Names of VGG layers from which to compute the style loss.

    content_layer_names : list of strings
        Names of VGG layers from which to compute the content loss.

    content_weight : float
        Weight of the content term in the optimization function.  The
        style term computes the sum of the L2 norm between the Gram
        matrices of the different layers (using ImageNet-trained VGG)
        of the content and output images.

    verbose : boolean
        Print progress to the screen.

    Returns
    -------
    ANTs 3-component image.

    Example
    -------
    >>> image = neural_style_transfer(content_image, style_image)
    """

    def preprocess_ants_image(image, doScaleAndCenter=True):
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

        if doScaleAndCenter == True:
            for i in range(3):
                array[:,:,i] = (array[:,:,i] - array[:,:,i].min()) / (array[:,:,i].max() - array[:,:,i].min())
            array *= 255.0
            array -= 127.5

        array = np.expand_dims(array, 0)
        return(array)

    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return(gram)

    def style_loss(style_features, combination_features, image_shape):
        style_gram = gram_matrix(style_features)
        content_gram = gram_matrix(combination_features)
        size = image_shape[0] * image_shape[1]
        number_of_channels = image_shape[2]
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

    def compute_total_loss(content_array, style_array, combination_tensor, feature_model, content_layer_names, style_layer_names, image_shape):
        input_tensor = tf.concat([content_array, style_array, combination_tensor], axis=0)
        features = feature_model(input_tensor)

        total_loss = tf.zeros(shape=())

        # content loss
        for layer_name in content_layer_names:
            layer_features = features[layer_name]
            content_features = layer_features[0,:, :, :]
            combination_features = layer_features[2, :, :, :]
            total_loss = total_loss + content_weight * content_loss(content_features, combination_features)

        # style loss
        for layer_name in style_layer_names:
            layer_features = features[layer_name]
            style_features = layer_features[1,:, :, :]
            combination_features = layer_features[2, :, :, :]
            total_loss = total_loss + ((style_weight / len(style_layer_names))
                * style_loss(style_features, combination_features, image_shape))

        # total variation loss
        total_loss += total_variation_weight * total_variation_loss(combination_tensor)

        return(total_loss)


    if style_image.dimension != 2 or content_image.dimension != 2:
        raise ValueError("Input images must be 2-D.")

    model = vgg19.VGG19(weights="imagenet", include_top=False)

    outputs_dictionary = dict([(layer.name, layer.output) for layer in model.layers])

    feature_model = tf.keras.Model(inputs=model.inputs, outputs=outputs_dictionary)

    # Preprocess data
    style_array = preprocess_ants_image(style_image)
    content_array = preprocess_ants_image(content_image)

    image_shape = (content_array.shape[1], content_array.shape[2], 3)

    combination_tensor = None
    if initial_combination_image is None:
        combination_tensor = tf.Variable(np.copy(content_array))
    else:
        initial_combination_tensor = preprocess_ants_image(initial_combination_image, doScaleAndCenter=False)
        combination_tensor = tf.Variable(initial_combination_tensor)

    if not image_shape == (combination_tensor.shape[1], combination_tensor.shape[2], 3):
        raise ValueError("Initial combination image size does not match content image.")

    # Add a tf.function decorator to loss & gradient computation
    # to compile it, and thus make it fast.

    @tf.function
    def compute_loss_and_gradients(content_array, style_array, combination_tensor,
      feature_model, content_layer_names, style_layer_names, image_shape):
        with tf.GradientTape() as tape:
            loss = compute_total_loss(content_array, style_array, combination_tensor,
                                      feature_model, content_layer_names,
                                      style_layer_names, image_shape)
        gradients = tape.gradient(loss, combination_tensor)
        return loss, gradients

    optimizer = tf.keras.optimizers.SGD(tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=10.0, decay_steps=100, decay_rate=0.96))

    for i in range(number_of_iterations):
        loss, gradients = compute_loss_and_gradients(content_array, style_array,
                              combination_tensor, feature_model, content_layer_names,
                              style_layer_names, image_shape)
        if verbose == True:
            print("Iteration %d: total loss = %.2f" % (i, loss))
        optimizer.apply_gradients([(gradients, combination_tensor)])

    combination_image = ants.from_numpy(np.squeeze(combination_tensor.numpy()),
        origin=content_image.origin, spacing=content_image.spacing,
        direction=content_image.direction, has_components=True)

    return(combination_image)
