Utilities
===================================

Custom loss functions
---------------------

.. autofunction:: antspynet.utilities.multilabel_dice_coefficient
.. autofunction:: antspynet.utilities.loss_multilabel_dice_coefficient_error

.. autofunction:: antspynet.utilities.peak_signal_to_noise_ratio
.. autofunction:: antspynet.utilities.loss_peak_signal_to_noise_ratio_error

.. autofunction:: antspynet.utilities.pearson_correlation_coefficient
.. autofunction:: antspynet.utilities.loss_pearson_correlation_coefficient_error

Patches
-------

.. autofunction:: antspynet.utilities.extract_image_patches
.. autofunction:: antspynet.utilities.reconstruct_image_from_patches

Clustering
----------

.. autoclass:: antspynet.utilities.Clustering

.. autoclass:: antspynet.utilities.DeepEmbeddedClusteringModel

Dense Unet
----------

.. autoclass:: antspynet.utilities.Scale

Mixture density
---------------

.. autoclass:: antspynet.utilities.MixtureDensityLayer

.. autofunction:: antspynet.utilities.get_mixture_density_loss_function
.. autofunction:: antspynet.utilities.get_mixture_density_sampling_function
.. autofunction:: antspynet.utilities.get_mixture_density_mse_function
.. autofunction:: antspynet.utilities.split_mixture_parameters
.. autofunction:: antspynet.utilities.mixture_density_software_max
.. autofunction:: antspynet.utilities.sample_from_output


Super-resolution
----------------

.. autofunction:: antspynet.utilities.mse
.. autofunction:: antspynet.utilities.mae
.. autofunction:: antspynet.utilities.psnr
.. autofunction:: antspynet.utilities.ssim
.. autofunction:: antspynet.utilities.gmsd

Spatial transformer network
---------------------------

.. autoclass:: antspynet.utilities.SpatialTransformer2D
.. autoclass:: antspynet.utilities.SpatialTransformer3D

Misc.
-----

.. autofunction:: antspynet.utilities.get_pretrained_network









