[![Build Status](https://travis-ci.org/ANTsX/ANTsPyNet.svg?branch=master)](https://travis-ci.org/ANTsX/ANTsPyNet)

# ANTsPyNet

A collection of deep learning architectures ported to the python language and tools for basic medical image processing. Based on `keras` and `tensorflow` with cross-compatibility with our R analog [ANTsRNet](https://github.com/ANTsX/ANTsRNet/).

Applications available at [ANTsXNet Apps](https://github.com/ANTsXNet).

Documentation page [https://antsx.github.io/ANTsPyNet/](https://antsx.github.io/ANTsPyNet/).

## Architectures

### Image voxelwise segmentation/regression

* U-Net (2-D, 3-D)
    * [O. Ronneberger, P. Fischer, and T. Brox.  U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)
* U-Net + ResNet (2-D, 3-D)
    * [Michal Drozdzal, Eugene Vorontsov, Gabriel Chartrand, Samuel Kadoury, and Chris Pal.  The Importance of Skip Connections in Biomedical Image Segmentation.](https://arxiv.org/abs/1608.04117)
* Dense U-Net (2-D, 3-D)
    * [X. Li, H. Chen, X. Qi, Q. Dou, C.-W. Fu, P.-A. Heng. H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes.](https://arxiv.org/pdf/1709.07330.pdf)    

### Image classification/regression

* AlexNet (2-D, 3-D)
    * [A. Krizhevsky, and I. Sutskever, and G. Hinton. ImageNet Classification with Deep Convolutional Neural Networks.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* VGG (2-D, 3-D)
    * [K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.](https://arxiv.org/abs/1409.1556)
* ResNet/ResNeXt (2-D, 3-D)
    * [Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.  Deep Residual Learning for Image Recognition.](https://arxiv.org/abs/1512.03385)
    * [Saining Xie and Ross Girshick and Piotr Dollár and Zhuowen Tu and Kaiming He.  Aggregated Residual Transformations for Deep Neural Networks.](https://arxiv.org/abs/1611.05431)
* WideResNet (2-D, 3-D)
    * [Sergey Zagoruyko and Nikos Komodakis.  Wide Residual Networks.](http://arxiv.org/abs/1605.07146)    
* DenseNet (2-D, 3-D)
    * [G. Huang, Z. Liu, K. Weinberger, and L. van der Maaten. Densely Connected Convolutional Networks Networks.](https://arxiv.org/abs/1608.06993)

### Object detection

### Image super-resolution

* Super-resolution convolutional neural network (SRCNN) (2-D, 3-D)
    * [Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang.  Image Super-Resolution Using Deep Convolutional Networks.](https://arxiv.org/abs/1501.00092)
* Expanded super-resolution (ESRCNN) (2-D, 3-D)
    * [Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang.  Image Super-Resolution Using Deep Convolutional Networks.](https://arxiv.org/abs/1501.00092)
* Denoising auto encoder super-resolution (DSRCNN) (2-D, 3-D)
* Deep denoise super-resolution (DDSRCNN) (2-D, 3-D)
    * [Xiao-Jiao Mao, Chunhua Shen, and Yu-Bin Yang.  Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections](https://arxiv.org/abs/1606.08921)
* ResNet super-resolution (SRResNet) (2-D, 3-D)
    * [Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, and Wenzhe Shi.  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.](https://arxiv.org/abs/1609.04802)    
* Deep back-projection network (DBPN) (2-D, 3-D)
    * [Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita.  Deep Back-Projection Networks For Super-Resolution.](https://arxiv.org/abs/1803.02735)
* Super resolution GAN
    * [Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi.  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.](https://arxiv.org/abs/1609.04802)
    
### Registration and transforms

* Spatial transformer network (STN) (2-D, 3-D)
    * [Max Jaderberg, Karen Simonyan, Andrew Zisserman, and Koray Kavukcuoglu.  Spatial Transformer Networks.](https://arxiv.org/abs/1506.02025)

### Generative adverserial networks

* Generative adverserial network (GAN)
    * [Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio.  Generative Adverserial Networks.](https://arxiv.org/abs/1406.2661)
* Deep Convolutional GAN
    * [Alec Radford, Luke Metz, Soumith Chintala.  Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.](https://arxiv.org/abs/1511.06434)
* Wasserstein GAN
    * [Martin Arjovsky, Soumith Chintala, Léon Bottou.  Wasserstein GAN.](https://arxiv.org/abs/1701.07875)
* Improved Wasserstein GAN
    * [Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville.  Improved Training of Wasserstein GANs.](https://arxiv.org/abs/1704.00028)
* Cycle GAN
    * [Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.  Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.](https://arxiv.org/abs/1703.10593)
* Super resolution GAN
    * [Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi.  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.](https://arxiv.org/abs/1609.04802)
    
### Clustering

* Deep embedded clustering (DEC)
    * [Junyuan Xie, Ross Girshick, and Ali Farhadi.  Unsupervised Deep Embedding for Clustering Analysis.](https://arxiv.org/abs/1511.06335)
* Deep convolutional embedded clustering (DCEC)
    * [Xifeng Guo, Xinwang Liu, En Zhu, and Jianping Yin.  Deep Clustering with Convolutional Autoencoders.](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)

### Miscellaneous

* Mixture density networks (MDN)
    * [Christopher M. Bishop.  Mixture Density Networks.](https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf)

--------------------------------------

## Installation

* ANTsPyNet Installation:
    * Option 1:
       ```
       $ git clone https://github.com/ANTsX/ANTsPyNet
       $ cd ANTsPyNet
       $ python setup.py install
       ```

## Quick example

```
import numpy as np
import keras as ke
import ants
import antspynet
import antspynet.architectures as apa
import antspynet.utilities as apu
nChannels = 1
patchWidth = 32
pw2 = ( patchWidth, patchWidth )
image = ants.image_read(ants.get_ants_data('r16'))
image2 = ants.image_math( image, "Grad" )
image_patches = apu.extract_image_patches(image, patch_size = pw2,
  max_number_of_patches = 64, return_as_array = True, random_seed = 1 )
image2_patches = apu.extract_image_patches(image2, patch_size = pw2,
  max_number_of_patches = 64, return_as_array = True, random_seed = 1 )
xarray = np.zeros( [len( image_patches ), patchWidth, patchWidth, nChannels ] )
yarray = np.zeros( [len( image_patches ), patchWidth, patchWidth, nChannels ] )
for x in range( 0, len( image_patches ) ):
  xarray[x,:,:,0] = image_patches[x][:,:]
  yarray[x,:,:,0] = image2_patches[x][:,:]
##########################################
model = apa.create_unet_model_2d( ( None, None, nChannels ),
  number_of_layers = 4, mode = 'regression' )
model.summary()
model.compile( loss = ke.losses.mse,
               optimizer = ke.optimizers.Adam( lr = 0.0001 ) )
model.fit( xarray, yarray, epochs = 125, batch_size = 8 )
preds = model.predict( xarray )
k = 12
pimgIn = ants.from_numpy( xarray[k,:,:,0] )
pimgPred = ants.from_numpy( preds[k,:,:,0] )
pimgGT = ants.from_numpy( yarray[k,:,:,0] )
ants.plot( pimgIn )
ants.plot( pimgPred )
ants.plot( pimgGT )
```

compare to ANTsRNet

```
library( keras )
library( tensorflow )
library( ANTsR )
library( ANTsRNet )
nChannels = 1
patchWidth = 32
pw2 = c( patchWidth, patchWidth )
image = antsImageRead( getANTsRData( "r16" ) )
image_patches = extractImagePatches( image, patchSize = pw2,
  maxNumberOfPatches = 64, returnAsArray = TRUE, randomSeed = 1 )
image2 = iMath( image, "Grad" )
image2_patches = extractImagePatches( image2, patchSize = pw2,
  maxNumberOfPatches = 64, returnAsArray = TRUE, randomSeed = 1  )
xarray = array( dim = c( nrow( image_patches ), patchWidth, patchWidth, nChannels ) )
yarray = array( dim = c( nrow( image_patches ), patchWidth, patchWidth, nChannels ) )
for ( x in 1:nrow( image_patches ) ) {
  xarray[x,,,1] = image_patches[x,,]
  yarray[x,,,1] = image2_patches[x,,]
  }
modelR = createUnetModel2D( list( NULL, NULL, nChannels ), numberOfLayers = 2,
  mode = 'regression' )
modelR %>% compile( loss = 'mse',
               optimizer = optimizer_adam( lr = 1e-2 ) )
modelR %>% fit( xarray, yarray, epochs = 10, batch_size = 8 )
preds = modelR %>% predict( xarray )
k = 10
pimgIn = as.antsImage( xarray[k,,,1] )
pimgPred = as.antsImage( preds[k,,,1] )
pimgGT = as.antsImage( yarray[k,,,1] )
layout( matrix( 1:3, nrow=1 ))
plot( pimgIn , colorbar = FALSE )
plot( pimgPred , colorbar = FALSE  )
plot( pimgGT , colorbar = FALSE  )
```


## Publications

* Nicholas J. Tustison, Brian B. Avants, and James C. Gee. Learning image-based spatial transformations via convolutional neural networks: a review,  _Magnetic Resonance Imaging_.  [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/31200026)

* Nicholas J. Tustison, Brian B. Avants, Zixuan Lin, Xue Feng, Nicholas Cullen, Jaime F. Mata, Lucia Flors, James C. Gee, Talissa A. Altes, John P. Mugler III, and Kun Qing.  Convolutional Neural Networks with Template-Based Data Augmentation for Functional Lung Image Quantification, _Academic Radiology_, 26(3):412-423, Mar 2019. [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30195415)

* Andrew T. Grainger, Nicholas J. Tustison, Kun Qing, Rene Roy, Stuart S. Berr, and Weibin Shi.  Deep learning-based quantification of abdominal fat on magnetic resonance images. _PLoS One_, 13(9):e0204071, Sep 2018.  [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30235253)

* Cullen N.C., Avants B.B. (2018) Convolutional Neural Networks for Rapid and Simultaneous Brain Extraction and Tissue Segmentation. In: Spalletta G., Piras F., Gili T. (eds) Brain Morphometry. Neuromethods, vol 136. Humana Press, New York, NY [doi](https://doi.org/10.1007/978-1-4939-7647-8_2)

## Acknowledgments

* We gratefully acknowledge the support of the NVIDIA Corporation with the donation of two Titan Xp GPUs used for this research.

* We gratefully acknowledge the grant support of the [Office of Naval Research](https://www.onr.navy.mil) and [Cohen Veterans Bioscience](https://www.cohenveteransbioscience.org).
