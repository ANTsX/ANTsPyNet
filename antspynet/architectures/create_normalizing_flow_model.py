import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def create_normalizing_flow_model(input_size,
                                  mask=None,
                                  hidden_layers=[256, 256],
                                  flow_steps=4,
                                  regularization=0.01,
                                  validate_args=False):
    """
    
    Normalizing flow model.  Taken from https://github.com/aganse/flow_models.

    Arguments
    ---------
    input_size : tuple or int
        If tuple, specifies the input size of the image.  If int, specifies the input vector size.
        
    mask : ANTsImage (optional)
        Specifies foreground.    

    number_of_hidden__layer : integer
        number of filters at the initial dense layer.  This number is halved for
        each subsequent layer.

    number_of_layers : integer
        Number of dense layers defining the model.

    mode : string
        "regression" or "classification".

    number_of_outputs : integer
        Specifies output for networks.

    Returns
    -------
    Keras model
        A Keras model defining the network.

    """

    class FlowModel(tf.keras.Model):
        
        def __init__(self,
                     input_length,
                     hidden_layers,
                     flow_steps,
                     regularization,
                     validate_args):

            super().__init__()
            self.input_length = input_length
            
            base_layer_name = "flow_step"
            flow_step_list = []
            for i in range(flow_steps):
                flow_step_list.append(tfp.bijectors.BatchNormalization(
                    validate_args=validate_args,
                    name="{}_{}/batchnorm".format(base_layer_name, i)))
                flow_step_list.append(tfp.bijectors.Permute(
                    permutation=list(np.random.permutation(input_length)),
                    validate_args=validate_args,
                    name="{}_{}/permute".format(base_layer_name, i)))
                flow_step_list.append(tfp.bijectors.RealNVP(
                    num_masked=input_length // 2,
                    shift_and_log_scale_fn=tfp.bijectors.real_nvp_default_template(
                        hidden_layers=hidden_layers,
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=0),
                        kernel_regularizer=tf.keras.regularizers.l2(regularization)),
                    validate_args=validate_args,
                    name="{}_{}/realnvp".format(base_layer_name, i)))
                
                # Remove last permutation
                flow_step_list = list(flow_step_list[:])
                
                self.flow_bijector_chain = tfp.bijectors.Chain(flow_step_list,
                                                               validate_args=validate_args,
                                                               name=base_layer_name)

                base_distribution = tfp.distributions.MultivariateNormalDiag(loc=[0.0] * input_length)  
                self.flow = tfp.distributions.TransformedDistribution(
                    distribution=base_distribution,
                    bijector=self.flow_bijector_chain,
                    name="top_level_flow_model")

        @tf.function
        def call(self, inputs):
            # images to gaussian points
            return self.flow.bijector.forward(inputs)
        
        @tf.function
        def inverse(self, outputs):
            # gaussian points to image
            return self.flow.bijector.inverse(outputs)        

        @tf.function
        def train_step(self, data):
            with tf.GradientTape() as tape:
                log_probability = self.flow.log_prob(data)
                if (tf.reduce_any(tf.math.is_nan(log_probability)) or 
                    tf.reduce_any(tf.math.is_inf(log_probability))):
                    tf.print("NaN or Inf detected in log_probability.")
                negative_log_likelihood = -tf.reduce_mean(log_probability)
                gradients = tape.gradient(negative_log_likelihood, self.flow.trainable_variables)
                if tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)) for g in gradients]):
                    tf.print("NaN or Inf detected in gradients.")
                gradients = [tf.clip_by_value(g, -1.0, 1.0) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
            bits_per_dimension_divisor = np.prod(self.image_shape) * tf.math.log(2.0)
            bpd = negative_log_likelihood / bits_per_dimension_divisor
            return {"neg_log_likelihood": negative_log_likelihood,
                    "bits_per_dim": bpd}
        

    if isinstance(input_size, int):
        input_length = input_size
    else:    
        flattened_image_size = np.prod(input_size)
        if mask is not None:
            number_of_channels = input_size[-1]
            flattened_image_size = len(mask[mask > 0]) * number_of_channels
        input_length = flattened_image_size    

    model = FlowModel(input_length=input_length,
                      hidden_layers=hidden_layers,
                      flow_steps=flow_steps,
                      regularization=regularization,
                      validate_args=validate_args)
    model.build((None, *(input_length,)))

    return(model)