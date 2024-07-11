import numpy as np
from astropy.stats import sigma_clip


def _sigma_mask(data, sigma=3):
    """
    Just does a sigma clip on an array.

    Parameters
    ----------
    data : array_like
            A single image

    sigma : float
            sigma used in the sigma clipping

    Returns
    -------
    clipped : array_like
            A boolean array to mask the original array
    """
    clipped = ~sigma_clip(data, sigma=sigma).mask
    return clipped


def _strip_units(data):
    """
    Removes the units off of data that was not in a NDarray, such as an astropy table. Returns an NDarray that has no units 

    Parameters:
    ----------
    data: array_like
            array_like set of data that may have associated units that want to be removed. Should be able to return something sensible when .values is called.

    Returns:
    -------
    data: array_like
            Same shape as input data, but will not have any units
    """
    if type(data) != np.ndarray:
        data = data.value
    return data


def grads_rad(flux):
    """
    Calculates the radius of the flux from the gradient of the flux, and the double gradient of the flux.  

    Parameters:
    ----------
    flux: array_like
            An array of flux values

    Returns:
    -------
    rad: array_like
            The radius of the fluxes 
    """
    rad = np.sqrt(np.gradient(flux)**2+np.gradient(np.gradient(flux))**2)
    return rad


def grad_flux_rad(flux):
    """
    Calculates the radius of the flux from the gradient of the flux.  

    Parameters:
    ----------
    flux: array_like
            An array of flux values

    Returns:
    -------
    rad: array_like
            The radius of the fluxes 
    """
    rad = np.sqrt(flux**2+np.gradient(flux)**2)
    return rad
