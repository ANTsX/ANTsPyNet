import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Add, Activation, AveragePooling3D, BatchNormalization,
                                     Conv3D, Dropout, Input, MaxPooling3D, ReLU, ZeroPadding3D)
from tensorflow.keras.layers import Input, Dropout, Concatenate, Multiply, Lambda, Add
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D,MaxPooling2D,Conv2DTranspose

from antspynet.utilities import LogSoftmax                          

def create_simple_fully_convolutional_network_model_3d(input_image_size,
                                                       number_of_filters_per_layer=(32, 64, 128, 256, 256, 64),
                                                       number_of_bins=40,
                                                       dropout_rate=0.5):
    """
    Implementation of the "SCFN" architecture for Brain/Gender prediction

    Creates a keras model implementation of the Simple Fully Convolutional
    Network model from the FMRIB group:

       https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain


    Arguments
    ---------
    input_image_size : tuple of length 4
        Used for specifying the input tensor shape.  The shape (or dimension) of
        that tensor is the image dimensions followed by the number of channels
        (e.g., red, green, and blue).
    number_of_filters_per_layer : array 
        number of filters for the convolutional layers.
    number_of_bins : integer
        number of bins for final softmax output.
    dropout_rate : float between 0 and 1
        Optional dropout rate before final convolution layer. 

    Returns
    -------
    Keras model
        A 3-D keras model.

    Example
    -------
    >>> model = create_simple_fully_convolutional_network_model_3d((None, None, None, 1))
    >>> model.summary()
    """

    number_of_layers = len(number_of_filters_per_layer)

    inputs = Input(shape=input_image_size)
 
    outputs = inputs
    for i in range(number_of_layers):
        if i < number_of_layers - 1:
            outputs = Conv3D(filters=number_of_filters_per_layer[i],
                             kernel_size=(3, 3, 3),
                             padding='valid')(outputs)
            outputs = ZeroPadding3D(padding=(1, 1, 1))(outputs)                 
            outputs = BatchNormalization(momentum=0.1,
                                         epsilon=1e-5)(outputs)
            outputs = MaxPooling3D(pool_size=(2, 2, 2),
                                   strides=(2, 2, 2))(outputs)
        else:
            outputs = Conv3D(filters=number_of_filters_per_layer[i],
                             kernel_size=(1, 1, 1),
                             padding='valid')(outputs)
            outputs = BatchNormalization(momentum=0.1,
                                         epsilon=1e-5)(outputs)
        outputs = ReLU()(outputs)

    outputs = AveragePooling3D(pool_size=(5, 6, 5),
                               strides=(5, 6, 5))(outputs)

    if dropout_rate > 0.0:
        outputs = Dropout(rate=dropout_rate)(outputs)

    outputs = Conv3D(filters=number_of_bins,
                     kernel_size=(1, 1, 1),
                     padding='valid')(outputs)
    outputs = LogSoftmax()(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def create_rmnet_generator():
    
    """
    Implementation of the "RMNet" generator architecture for inpainting

    Creates a keras model implementation of the model:

       https://github.com/Jireh-Jam/R-MNet-Inpainting-keras

    Returns
    -------
    Keras model
        A 3-D keras model.

    Example
    -------
    >>> model = create_rmnet_model()
    >>> model.summary()
    """

    def reverse_mask(x):
        return 1-x

    img_shape = (256, 256, 3)
    img_shape_mask = (256, 256, 1) 
    gf = 64
    channels = 3

    #compute inputs
    input_img = Input(shape=img_shape, dtype='float32', name='image_input')
    input_mask = Input(shape=img_shape_mask, dtype='float32',name='mask_input')  
    
    reversed_mask = Lambda(reverse_mask,output_shape=(img_shape_mask))(input_mask)
    masked_image = Multiply()([input_img,reversed_mask])
    
    #encoder
    x =(Conv2D(gf,(5, 5), dilation_rate=2, input_shape=img_shape, padding="same",name="enc_conv_1"))(masked_image)
    x =(LeakyReLU(alpha=0.2))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    pool_1 = MaxPooling2D(pool_size=(2,2))(x) 
    
    x =(Conv2D(gf,(5, 5), dilation_rate=2, padding="same",name="enc_conv_2"))(pool_1)
    x =(LeakyReLU(alpha=0.2))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    pool_2 = MaxPooling2D(pool_size=(2,2))(x) 
    
    x =(Conv2D(gf*2, (5, 5), dilation_rate=2, padding="same",name="enc_conv_3"))(pool_2)
    x =(LeakyReLU(alpha=0.2))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    pool_3 = MaxPooling2D(pool_size=(2,2))(x) 
    
    x =(Conv2D(gf*4, (5, 5), dilation_rate=2, padding="same",name="enc_conv_4"))(pool_3)
    x =(LeakyReLU(alpha=0.2))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    pool_4 = MaxPooling2D(pool_size=(2,2))(x) 
    
    x =(Conv2D(gf*8, (5, 5), dilation_rate=2, padding="same",name="enc_conv_5"))(pool_4)
    x =(LeakyReLU(alpha=0.2))(x)
    x =(Dropout(0.5))(x)
    
    #Decoder
    x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
    x =(Conv2DTranspose(gf*8, (3, 3), padding="same",name="upsample_conv_1"))(x)
    x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
    x =(Activation('relu'))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
    x = (Conv2DTranspose(gf*4, (3, 3),  padding="same",name="upsample_conv_2"))(x)
    x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
    x =(Activation('relu'))(x)
    x =(BatchNormalization(momentum=0.8))(x)
        
    x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
    x = (Conv2DTranspose(gf*2, (3, 3),  padding="same",name="upsample_conv_3"))(x)
    x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
    x =(Activation('relu'))(x)
    x =(BatchNormalization(momentum=0.8))(x)
        
    x =(UpSampling2D(size=(2, 2), interpolation='bilinear'))(x)
    x = (Conv2DTranspose(gf, (3, 3),  padding="same",name="upsample_conv_4"))(x)
    x = Lambda(lambda x: tf.pad(x,[[0,0],[0,0],[0,0],[0,0]],'REFLECT'))(x)
    x =(Activation('relu'))(x)
    x =(BatchNormalization(momentum=0.8))(x)
    
    x = (Conv2DTranspose(channels, (3, 3),  padding="same",name="final_output"))(x)
    x =(Activation('tanh'))(x)

    decoded_output = x
    reversed_mask_image = Multiply()([decoded_output, input_mask])
    output_img = Add()([masked_image,reversed_mask_image])
    concat_output_img = Concatenate()([output_img,input_mask])
    model = Model(inputs = [input_img, input_mask], outputs = [concat_output_img])

    return model 
