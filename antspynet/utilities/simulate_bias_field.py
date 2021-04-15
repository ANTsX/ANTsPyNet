import ants

import numpy as np

def simulate_bias_field(domain_image,
                        number_of_points=10,
                        sd_bias_field=1.0,
                        number_of_fitting_levels=4,
                        mesh_size=1):
    """
    Simulate random bias field

    Low frequency, spatial varying simulated random bias field using
    random points and B-spline fitting.

    Arguments
    ---------
    domain_image : ANTsImage
        Image to define the spatial domain of the bias field.

    number_of_points : integer
        Number of randomly defined points to define the bias field
        (default = 10).

    sd_bias_field : float
        Characterize the standard deviation of the amplitude (default = 1).

    number_of_fitting_levels : integer
        B-spline fitting parameter.

    clamp_end_points : integer or tuple
        B-spline fitting parameter.

    Returns
    -------
    ANTs image

    Example
    -------
    >>> import ants
    >>> image = ants.image_read(ants.get_ants_data("r64"))
    >>> bias_field_image = simulate_bias_field( image )
    """

    dimension = domain_image.dimension
    origin = ants.get_origin(domain_image)
    spacing = ants.get_spacing(domain_image)
    direction = ants.get_direction(domain_image)
    shape = domain_image.shape

    min_spatial_domain = origin
    max_spatial_domain = origin + (np.array(shape) - 1.0) * spacing

    scattered_data = np.zeros((number_of_points,1))
    parametric_data = np.zeros((number_of_points, dimension))

    scattered_data[:,0] = np.random.normal(loc=0.0, scale=sd_bias_field, size=number_of_points)
    for d in range(dimension):
        parametric_data[:,d] = np.random.uniform(low=min_spatial_domain[d],
            high=max_spatial_domain[d], size=number_of_points)

    if isinstance(mesh_size, int):
        mesh_size = np.tile(mesh_size, dimension)

    bias_field = ants.fit_bspline_object_to_scattered_data(scattered_data,
        parametric_data, parametric_domain_origin=origin, parametric_domain_spacing=spacing,
        parametric_domain_size=shape, number_of_fitting_levels=number_of_fitting_levels,
        mesh_size=mesh_size)

    ants.set_direction(bias_field, direction)

    return bias_field