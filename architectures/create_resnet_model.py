
from keras.models import Model
from keras.layers import (Input, Dropout, BatchNormalization, Add,
                          LeakyReLU, Concatenate, Lambda, Dense,
                          Conv2D, Conv2DTranspose,
                          MaxPooling2D, GlobalAveragePooling2D,
                          UpSampling2D,
                          Conv3D, Conv3DTranspose,
                          MaxPooling3D, GlobalAveragePooling3D,
                          UpSampling3D)

def create_resnet_model_2d(input_image_size,
                           number_of_classification_labels=1000,
                           layers=(1, 2, 3, 4),
                           residual_block_schedule=(3, 4, 6, 3),
                           lowest_resolution=64,
                           cardinality=1,
                           mode='classification'
                          ):
    """
    2-D implementation of the ResNet deep learning architecture.

    Creates a keras model of the ResNet deep learning architecture for image
    classification.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

    :param input_image_size: Used for specifying the input tensor shape.  The
      shape (or dimension) of that tensor is the image dimensions followed by
      the number of channels (e.g., red, green, and blue).  The batch size
      (i.e., number of training images) is not specified a priori.
    :param number_of_classification_labels: Number of segmentation labels.
    :param layers: a array determining the number of 'filters' defined at
      for each layer.
    :param residual_block_schedule: array defining the how many residual blocks
      repeats.
    :param lowest_resolution: number of filters at the initial layer.
    :param cardinality: perform ResNet (cardinality = 1) or ResNeXt
      (cardinality != 1 but powers of 2---try '32')
    :param mode: 'classification' or 'regression'.  Default = 'classification'.
    :returns: an ResNet keras model.
    """

    def add_common_layers(model):
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        return(model)

    def grouped_convolution_layer_2d(model, number_of_filters, strides):
        # Per standard ResNet, this is just a 2-D convolution
        if cardinality == 1:
            grouped_model = Conv2D(filters=number_of_filters,
                                   kernel_size=(3, 3),
                                   strides=strides,
                                   padding='same')(model)
            return(grouped_model)

        if number_of_filters % cardinality != 0:
            raise ValueError('number_of_filters `%` cardinality != 0')

        number_of_group_filters = int(number_of_filters / cardinality)

        convolution_layers = []
        for j in range(cardinality):
            local_layer = Lambda(lambda z: z[:, :, :,
              j * number_of_group_filters:j * number_of_group_filters + number_of_group_filters])(model)
            convolution_layers.append(Conv2D(filters=number_of_group_filters,
                                             kernel_size=(3, 3),
                                             strides=strides,
                                             padding='same')(local_layer))

        grouped_model = Concatenate()(convolution_layers)
        return(grouped_model)

    def residual_block_2d(model, number_of_filters_in, number_of_filters_out, strides=(1, 1), project_shortcut=False):
        shortcut = model

        model = Conv2D(filters=number_of_filters_in,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(model)
        model = add_common_layers(model)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        model = grouped_convolution_layer_2d(model,
                                             number_of_filters=number_of_filters_in,
                                             strides=strides)
        model = add_common_layers(model)

        model = Conv2D(filters=number_of_filters_out,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       padding='same')(model)
        model = BatchNormalization()(model)

        if project_shortcut == True or strides != (1,1):
            shortcut = Conv2D(filters=number_of_filters_out,
                              kernel_size=(1, 1),
                              strides=strides,
                              padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        model = Add()([shortcut, model])
        model = LeakyReLU()(model)
        return(model)


    inputs = Input(shape = input_image_size)

    n_filters = lowest_resolution

    outputs = Conv2D(filters=n_filters,
                     kernel_size=(7, 7),
                     strides=(2, 2),
                     padding='same')(inputs)
    outputs = add_common_layers(outputs)
    outputs = MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='same')(outputs)

    for i in range(len(layers)):
        n_filters_in = lowest_resolution * 2**layers[i]
        n_filters_out = 2 * n_filters_in

        for j in range(residual_block_schedule[i]):
            project_shortcut = False
            if i == 0 and j == 0:
                project_shortcut = True

            if i > 0 and j == 0:
                strides = (2, 2)
            else:
                strides = (1, 1)

            outputs = residual_block_2d(outputs,
                                        number_of_filters_in=n_filters_in,
                                        number_of_filters_out=n_filters_out,
                                        strides=strides,
                                        project_shortcut=project_shortcut)

    outputs = GlobalAveragePooling2D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        if number_of_classification_labels == 2:
            layer_activation = 'sigmoid'
        else:
            layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation)(outputs)

    resnet_model = Model(inputs=inputs, outputs=outputs)

    return(resnet_model)


def create_resnet_model_3d(input_image_size,
                           number_of_classification_labels=1000,
                           layers=(1, 2, 3, 4),
                           residual_block_schedule=(3, 4, 6, 3),
                           lowest_resolution=64,
                           cardinality=1,
                           mode='classification'
                          ):
    """
    3-D implementation of the ResNet deep learning architecture.

    Creates a keras model of the ResNet deep learning architecture for image
    classification.  The paper is available here:

            https://arxiv.org/abs/1512.03385

    This particular implementation was influenced by the following python
    implementation:

            https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

    :param input_image_size: Used for specifying the input tensor shape.  The
      shape (or dimension) of that tensor is the image dimensions followed by
      the number of channels (e.g., red, green, and blue).  The batch size
      (i.e., number of training images) is not specified a priori.
    :param number_of_classification_labels: Number of segmentation labels.
    :param layers: a array determining the number of 'filters' defined at
      for each layer.
    :param residual_block_schedule: array defining the how many residual blocks
      repeats.
    :param lowest_resolution: number of filters at the initial layer.
    :param cardinality: perform ResNet (cardinality = 1) or ResNeXt
      (cardinality != 1 but powers of 2---try '32')
    :param mode: 'classification' or 'regression'.  Default = 'classification'.
    :returns: an ResNet keras model.
    """

    def add_common_layers(model):
        model = BatchNormalization()(model)
        model = LeakyReLU()(model)
        return(model)

    def grouped_convolution_layer_3d(model, number_of_filters, strides):
        # Per standard ResNet, this is just a 3-D convolution
        if cardinality == 1:
            grouped_model = Conv3D(filters=number_of_filters,
                                   kernel_size=(3, 3, 3),
                                   strides=strides,
                                   padding='same')(model)
            return(grouped_model)

        if number_of_filters % cardinality != 0:
            raise ValueError('number_of_filters `%` cardinality != 0')

        number_of_group_filters = int(number_of_filters / cardinality)

        convolution_layers = []
        for j in range(cardinality):
            local_layer = Lambda(lambda z: z[:, :, :,
              j * number_of_group_filters:j * number_of_group_filters + number_of_group_filters])(model)
            convolution_layers.append(Conv3D(filters=number_of_group_filters,
                                             kernel_size=(3, 3, 3),
                                             strides=strides,
                                             padding='same')(local_layer))

        grouped_model = Concatenate()(convolution_layers)
        return(grouped_model)

    def residual_block_3d(model, number_of_filters_in, number_of_filters_out, strides=(1, 1, 1), project_shortcut=False):
        shortcut = model

        model = Conv3D(filters=number_of_filters_in,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1),
                       padding='same')(model)
        model = add_common_layers(model)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        model = grouped_convolution_layer_3d(model,
                                             number_of_filters=number_of_filters_in,
                                             strides=strides)
        model = add_common_layers(model)

        model = Conv3D(filters=number_of_filters_out,
                       kernel_size=(1, 1, 1),
                       strides=(1, 1, 1),
                       padding='same')(model)
        model = BatchNormalization()(model)

        if project_shortcut == True or strides != (1,1):
            shortcut = Conv3D(filters=number_of_filters_out,
                              kernel_size=(1, 1, 1),
                              strides=strides,
                              padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        model = Add()([shortcut, model])
        model = LeakyReLU()(model)
        return(model)


    inputs = Input(shape = input_image_size)

    n_filters = lowest_resolution

    outputs = Conv3D(filters=n_filters,
                     kernel_size=(7, 7, 7),
                     strides=(2, 2, 2),
                     padding='same')(inputs)
    outputs = add_common_layers(outputs)
    outputs = MaxPooling3D(pool_size=(3, 3, 3),
                           strides=(2, 2, 2),
                           padding='same')(outputs)

    for i in range(len(layers)):
        n_filters_in = lowest_resolution * 2**layers[i]
        n_filters_out = 2 * n_filters_in

        for j in range(residual_block_schedule[i]):
            project_shortcut = False
            if i == 0 and j == 0:
                project_shortcut = True

            if i > 0 and j == 0:
                strides = (2, 2, 2)
            else:
                strides = (1, 1, 1)

            outputs = residual_block_3d(outputs,
                                        number_of_filters_in=n_filters_in,
                                        number_of_filters_out=n_filters_out,
                                        strides=strides,
                                        project_shortcut=project_shortcut)


    outputs = GlobalAveragePooling3D()(outputs)

    layer_activation = ''
    if mode == 'classification':
        if number_of_classification_labels == 2:
            layer_activation = 'sigmoid'
        else:
            layer_activation = 'softmax'
    elif mode == 'regression':
        layerActivation = 'linear'
    else:
        raise ValueError('mode must be either `classification` or `regression`.')

    outputs = Dense(units=number_of_classification_labels,
                    activation=layer_activation)(outputs)

    resnet_model = Model(inputs=inputs, outputs=outputs)

    return(resnet_model)



