import numpy as np


def _sigma_mask(data,sigma=3):
	"""
	Just does a sigma clip on an array.

	Parameters
	----------
	data : array
		A single image 

	sigma : float
		sigma used in the sigma clipping

	Returns
	-------
	clipped : array
		A boolean array to mask the original array
	"""
	clipped = ~sigma_clip(data,sigma=sigma).mask
	return clipped 


def _strip_units(data):
	if type(data) != np.ndarray:
		data = data.value
	return data



def grads_rad(flux):
    rad = np.sqrt(np.gradient(flux)**2+np.gradient(np.gradient(flux))**2)
    return rad

def grad_flux_rad(flux):
    rad = np.sqrt(flux**2+np.gradient(flux)**2)
    return rad