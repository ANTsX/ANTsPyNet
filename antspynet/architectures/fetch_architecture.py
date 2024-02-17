
from inspect import getmembers, isfunction

from .. import architectures


def fetch_architecture(name, dim=None):
    """
    Fetch an architecture function based on its name and input image 
    dimensions (2, 3, or None).
    
    Arguments
    ---------

    name string
        One of the available architectures to be fetched. Available
        architectures are those with functions called 'create_{name}_model()'.
    
    dim integer (2 or 3)
        Whether to pull the 2-dimensional or 3-dimensional version of the architecture
        function. Can be left as None if there is no 2d vs 3d model (e.g., for autoencoders)

    Returns
    -------
    A filename string

    Example
    -------
    >>> from antspynet import fetch_architecture
    >>> vgg_fn = fetch_architecture('vgg', 3)
    >>> vgg_model = vgg_fn((128, 128, 128, 1))
    >>> autoencoder_fn = fetch_architecture('autoencoder')
    >>> autoencoder_model = autoencoder_fn((784, 500, 500, 2000, 10))
    """
    try:
        if dim is not None:
            fstr = f'create_{name}_model_{dim}d'
            fn = getattr(architectures, fstr)
        else:
            fstr = f'create_{name}_model'
            fn = getattr(architectures, fstr)
    except AttributeError:
        raise ValueError(f'Architecture function {fstr} does not exist.')
    return fn


def list_architectures():
    """
    List all available architectures and their context-of-use.
    
    Arguments
    ---------
    N/A
    
    Returns
    -------
    A list of strings where each string-item is the name of an 
    architecture that can be created with `create_{name}_model`.
    
    Example
    -------
    >>> from antspynet import list_architectures
    >>> archs = list_architectures()    
    """
    archs = [f[0].split('create_')[1].split('_model_') for f in getmembers(architectures, isfunction) if f[0].startswith('create_')]
    # add empty string for non-dimensioned models just to be consistent
    def add_empty(x):
        if len(x) == 1:
            x.append('')
        return x
    archs = [add_empty(arch) for arch in archs]
    return archs