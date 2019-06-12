
import numpy as np

import ants

def mse(x,
        y=None):
    """
    Mean square error of a single image or between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value.

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = mse(r16, r64)
    """

    if y == None:
       x2 = x**2
       return x2.mean()
    else:
        diff2 = (x - y) ** 2
        return diff2.mean()


def mae(x,
        y=None):
    """
    Mean absolute error of a single image or between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = mae(r16, r64)
    """

    if y == None:
       xabs = x.abs()
       return xabs.mean()
    else:
        diffabs = abs(x - y)
        return diffabs.mean()

def psnr(x,
         y):
    """
    Peak signal-to-noise ratio between two images.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = psnr(r16, r64)
    """

    value = (20 * np.log10(x.max()) - 10 * np.log10(mse(x, y)))
    return(value)

def ssim(x,
         y,
         K=(0.01, 0.03)):
    """
    Structural similarity index (SSI) between two images.

    Implementation of the SSI quantity for two images proposed in

    Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. "Image quality
    assessment: from error visibility to structural similarity". IEEE TIP.
    13 (4): 600–612.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    K : tuple of length 2
        tuple which contain SSI parameters meant to stabilize the formula
        in case of weak denominators.

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = psnr(r16, r64)
    """

    global_max <- np.max(x.max(), y.max())
    global_min <- np.abs(min(x.min(), y.min()))
    L = global_max - global_min

    C1 = (K[0] * L)**2
    C2 = (K[1] * L)**2
    C3 = C2 / 2

    mu_x <- x.mean()
    mu_y <- y.mean()

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = (x * x).mean() - mu_x_sq
    sigma_y_sq = (y * y).mean() - mu_y_sq
    sigma_xy = (x * y).mean() - mu_xy

    numerator = ( 2 * mu_xy + C1 ) * ( 2 * sigma_xy + C2 )
    denominator = ( mu_x_sq + mu_y_sq + C1 ) * ( sigma_x_sq + sigma_y_sq + C2 )

    SSI = numerator / denominator

    return(SSI)

def gmsd(x,
         y):
    """
    Gradient magnitude similarity deviation

    A fast and simple metric that correlates to perceptual quality.

    Arguments
    ---------
    x : input image
        ants input image

    y : input image
        ants input image

    Returns
    -------
    Value

    Example
    -------
    >>> r16 = ants.image_read(ants.get_data("r16"))
    >>> r64 = ants.image_read(ants.get_data("r64"))
    >>> value = gmsd(r16, r64)
    """

    gx = ants.iMath(x, "Grad")
    gy = ants.iMath(y, "Grad")

    # see eqn 4 - 6 in https://arxiv.org/pdf/1308.3052.pdf

    constant = 0.0026
    gmsd_numerator = gx * gy * 2.0 + constant
    gmsd_denominator = gx**2 + gy**2 + constant
    gmsd = gmsd_numerator / gmsd_denominator

    product_dimension = 1
    for i in range(len(x.shape)):
       product_dimension *= x.shape[i]
    prefactor = 1.0 / product_dimension

    return(np.sqrt(prefactor * ((gmsd - gmsd.mean())**2).sum()))
