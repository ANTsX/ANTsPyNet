# ANTsPyNet

A collection of deep learning architectures ported to the python language and tools for basic medical image processing. Based on `keras` and `tensorflow` with cross-compatibility with our R analog [ANTsRNet](https://github.com/ANTsX/ANTsRNet/).

Applications available at [ANTsRNet Apps](https://github.com/ANTsRNet).

## Architectures

### Image voxelwise segmentation/regression

* U-Net (2-D, 3-D)
    * [O. Ronneberger, P. Fischer, and T. Brox.  U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597)

### Image classification/regression

### Object detection

### Image super-resolution

### Registration and transforms

### Generative adverserial networks

### Clustering

### Miscellaneous

--------------------------------------

## Installation

* ANTsRNet Installation:
    * Option 1:
       ```
       $ R
       > devtools::install_github( "ANTsX/ANTsRNet" )
       ```
    * Option 2:
       ```
       $ git clone https://github.com/ANTsX/ANTsRNet.git
       $ R CMD INSTALL ANTsRNet
       ```

## Publications

* Nicholas J. Tustison, Brian B. Avants, and James C. Gee. Learning image-based spatial transformations via convolutional neural networks: a review,  _Magnetic Resonance Imaging_.  [(accepted)](https://bitbucket.org/ntustison/deepmap/src/master/Manuscript/stitched.pdf) 

* Nicholas J. Tustison, Brian B. Avants, Zixuan Lin, Xue Feng, Nicholas Cullen, Jaime F. Mata, Lucia Flors, James C. Gee, Talissa A. Altes, John P. Mugler III, and Kun Qing.  Convolutional Neural Networks with Template-Based Data Augmentation for Functional Lung Image Quantification, _Academic Radiology_, 26(3):412-423, Mar 2019. [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30195415)

* Andrew T. Grainger, Nicholas J. Tustison, Kun Qing, Rene Roy, Stuart S. Berr, and Weibin Shi.  Deep learning-based quantification of abdominal fat on magnetic resonance images. _PLoS One_, 13(9):e0204071, Sep 2018.  [(pubmed)](https://www.ncbi.nlm.nih.gov/pubmed/30235253)

* Cullen N.C., Avants B.B. (2018) Convolutional Neural Networks for Rapid and Simultaneous Brain Extraction and Tissue Segmentation. In: Spalletta G., Piras F., Gili T. (eds) Brain Morphometry. Neuromethods, vol 136. Humana Press, New York, NY [doi](https://doi.org/10.1007/978-1-4939-7647-8_2)

## Acknowledgments

* We gratefully acknowledge the support of the NVIDIA Corporation with the donation of two Titan Xp GPUs used for this research.

* We gratefully acknowledge the grant support of the [Office of Naval Research](https://www.onr.navy.mil) and [Cohen Veterans Bioscience](https://www.cohenveteransbioscience.org).
