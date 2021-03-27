
Utilities
=========

Custom metrics
--------------

.. autofunction:: antspynet.utilities.multilabel_dice_coefficient
.. autofunction:: antspynet.utilities.peak_signal_to_noise_ratio
.. autofunction:: antspynet.utilities.pearson_correlation_coefficient
.. autofunction:: antspynet.utilities.categorical_focal_loss
.. autofunction:: antspynet.utilities.weighted_categorical_crossentropy
.. autofunction:: antspynet.utilities.multilabel_surface_loss
.. autofunction:: antspynet.maximum_mean_discrepancy

Custom normalization layers
---------------------------

.. autoclass:: antspynet.utilities.InstanceNormalization

Custom activation layers
---------------------------

.. autoclass:: antspynet.utilities.LogSoftmax

Resample tensor layer
---------------------

.. autoclass:: antspynet.utilities.ResampleTensorLayer2D
.. autoclass:: antspynet.utilities.ResampleTensorLayer3D

Mixture density networks
------------------------

.. autoclass:: antspynet.utilities.MixtureDensityLayer
.. autofunction:: antspynet.utilities.get_mixture_density_loss_function

Attention
---------

.. autoclass:: antspynet.utilities.AttentionLayer2D
.. autoclass:: antspynet.utilities.AttentionLayer3D

Clustering
----------

.. autoclass:: antspynet.utilities.DeepEmbeddedClustering
.. autoclass:: antspynet.utilities.DeepEmbeddedClusteringModel

Image patch
-----------

.. autofunction:: antspynet.utilities.extract_image_patches
.. autofunction:: antspynet.utilities.reconstruct_image_from_patches

Super-resolution
-----------------

.. autofunction:: antspynet.utilities.mse
.. autofunction:: antspynet.utilities.mae
.. autofunction:: antspynet.utilities.psnr
.. autofunction:: antspynet.utilities.ssim
.. autofunction:: antspynet.utilities.gmsd
.. autofunction:: antspynet.utilities.apply_super_resolution_model_to_image

Spatial transformer network
---------------------------

.. autoclass:: antspynet.utilities.SpatialTransformer2D
.. autoclass:: antspynet.utilities.SpatialTransformer3D

Applications
------------

.. autofunction:: antspynet.utilities.brain_extraction
.. autofunction:: antspynet.utilities.cortical_thickness
.. autofunction:: antspynet.utilities.longitudinal_cortical_thickness
.. autofunction:: antspynet.utilities.lung_extraction
.. autofunction:: antspynet.utilities.preprocess_brain_image
.. autofunction:: antspynet.utilities.sysu_media_wmh_segmentation
.. autofunction:: antspynet.utilities.hippmapp3r_segmentation
.. autofunction:: antspynet.utilities.deep_flash
.. autofunction:: antspynet.utilities.deep_atropos
.. autofunction:: antspynet.utilities.desikan_killiany_tourville_labeling
.. autofunction:: antspynet.utilities.dkt_based_lobar_parcellation
.. autofunction:: antspynet.utilities.brain_age
.. autofunction:: antspynet.utilities.mri_super_resolution
.. autofunction:: antspynet.utilities.tid_neural_image_assessment
.. autofunction:: antspynet.utilities.neural_style_transfer

Miscellaneous
-------------

.. autofunction:: antspynet.utilities.get_pretrained_network
.. autofunction:: antspynet.utilities.get_antsxnet_data
.. autoclass:: antspynet.utilities.Scale
.. autofunction:: antspynet.utilities.regression_match_image
.. autofunction:: antspynet.utilities.randomly_transform_image_data
.. autofunction:: antspynet.utilities.histogram_warp_image_intensities
.. autofunction:: antspynet.utilities.crop_image_center
.. autofunction:: antspynet.utilities.pad_or_crop_image_to_size
.. autofunction:: antspynet.utilities.pad_image_by_factor
.. autofunction:: antspynet.utilities.encode_unet
.. autofunction:: antspynet.utilities.decode_unet
