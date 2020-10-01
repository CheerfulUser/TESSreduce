"""
Import packages!
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lightkurve as lk

from copy import deepcopy

from scipy.ndimage.filters import convolve
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter

from scipy.signal import savgol_filter

from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline

from photutils import centroid_com
from photutils import DAOStarFinder

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip

import multiprocessing
from joblib import Parallel, delayed

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
# set the package directory so we can load in a file later
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'


from astropy.coordinates import SkyCoord
from astropy import units as u

def Get_TESS(RA,DEC,Size=90,Sector=None):
	"""
	Use the lightcurve interface with TESScut to get an FFI cutout 
	of a region around the given coords.

	Parameters
	----------
	RA : float 
		RA of the centre point 

	DEC : float
		Dec of the centre point

	Size : int
		size of the cutout

	Sector : int
		sector to download 

	Returns
	-------
	tpf : lightkurve target pixel file
		tess ffi cutout of the selected region
	"""
	c = SkyCoord(ra=float(RA)*u.degree, dec=float(DEC) *
				 u.degree, frame='icrs')
	
	tess = lk.search_tesscut(c,sector=Sector)
	tpf = tess.download(cutout_size=Size)
	
	return tpf


def sigma_mask(data,sigma=3):
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


def Source_mask(Data, grid=0):
	"""
	Makes a mask of sources in the image using conditioning on percentiles.
	The grid option breakes the image up into sections the size of grid to
	do a basic median subtraction of the background. This is useful for 
	large fovs where the background has a lot of structure.

	Parameters
	----------
	data : array
		A single image 
	
	grid : int
		size of the averaging square used to do a median background subtraction 
		before finding the sources.

	Returns
	-------
	mask : array
		Boolean mask array for the sources in the image
	"""
	data = deepcopy(Data)
	if grid > 0:
		data[data<0] = np.nan
		data[data >= np.percentile(data,95)] =np.nan
		grid = np.zeros_like(data)
		size = grid
		for i in range(grid.shape[0]//size):
			for j in range(grid.shape[1]//size):
				section = data[i*size:(i+1)*size,j*size:(j+1)*size]
				section = section[np.isfinite(section)]
				lim = np.percentile(section,1)
				grid[i*size:(i+1)*size,j*size:(j+1)*size] = lim
		thing = data - grid
	else:
		thing = data
	ind = np.isfinite(thing)
	mask = ((thing <= np.percentile(thing[ind],80,axis=0)) |
		   (thing <= np.percentile(thing[ind],10))) * 1.0

	return mask

def Smooth_bkg(data, extrapolate = True):
	"""
	Interpolate over the masked objects to derive a background estimate. 

	Parameters
	----------
	data : array
		A single image 
	
	extrapolate: Bool
		switch for using extrapolation in the background 

	Returns
	-------
	estimate : array 
		an estimate of the smooth background in the TESS image

	bitmask : array
		an array indicating where extrapolation was used

	"""
	data[data == 0] = np.nan
	x = np.arange(0, data.shape[1])
	y = np.arange(0, data.shape[0])
	arr = np.ma.masked_invalid(data)
	xx, yy = np.meshgrid(x, y)
	#get only the valid values
	x1 = xx[~arr.mask]
	y1 = yy[~arr.mask]
	newarr = arr[~arr.mask]

	estimate = griddata((x1, y1), newarr.ravel(),
							  (xx, yy),method='linear')
	bitmask = np.zeros_like(data,dtype=int)
	bitmask[np.isnan(estimate)] = 128 | 4
	nearest = griddata((x1, y1), newarr.ravel(),
							  (xx, yy),method='nearest')
	if extrapolate:
		estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
	
	estimate = gaussian_filter(estimate,12)

	return estimate, bitmask
	
def Strap_bkg(Data):
	"""
	Calculate the additional background signal associated with the vertical detector straps

	Parameters
	----------
	Data : array
		A single masked image with only the strap regions preserved

	Returns:
	--------
	strap_bkg : array
		additional background from the detector straps 

	"""

	data = deepcopy(Data)
	data[data == 0] = np.nan
	strap_data = data[np.nansum(abs(data),axis=0)>0]
	source_mask = (data < np.percentile(strap_data[np.isfinite(strap_data)],70)) * 1.
	data = data * (source_mask == 1)
	data[data==0] = np.nan
	
	ind = np.where(np.nansum(abs(data),axis=0)>0)[0]
	strap_bkg = np.zeros_like(Data)
	for col in ind:
		x = np.arange(0,data.shape[1])
		y = data[:,col].copy()
		finite = np.isfinite(y)
		if len(y[finite]) > 5:
			finite = np.isfinite(y)
			bad = ~sigma_mask(y[finite],sigma=3)
			finite = np.where(finite)[0]
			y[finite[bad]] = np.nan
			finite = np.isfinite(y)
			
			if len(y[finite]) > 5:
				fit = UnivariateSpline(x[finite], y[finite])

				p = fit(x)
				finite = np.isfinite(p)
				smooth =savgol_filter(p[finite],13,3)
				p[finite] = smooth

				thingo = y - p
				finite = np.isfinite(thingo)
				bad = ~sigma_mask(thingo[finite],sigma=3)
				finite = np.where(finite)[0]
				y[finite[bad]] = np.nan
				finite = np.isfinite(y)
				
				if len(y[finite]) > 5:
					fit = UnivariateSpline(x[finite], y[finite])
					p = fit(x)
					finite = np.isfinite(p)
					smooth =savgol_filter(p[finite],13,3)
					p[finite] = smooth
					strap_bkg[:,col] = p

	return strap_bkg

def Calculate_bkg(data,straps,big_mask,big_strap):
	"""
	Function to calculate the background for a TESS tpf frame.

	Parameters
	----------
	data : array
		A single image 

	straps : array
		position of straps relative to the image 

	big_mask : array
		source mask convolved with a 3x3 kernal

	big_strap : array
		strap mask convolved with a 3x3 kernal

	Returns
	-------
	frame_bkg : array
		background estimate for a frame

	"""
	
	if np.nansum(data) > 0:
		masked = data * ((big_mask==0)*1) * ((big_strap==0)*1)
		masked[masked == 0] = np.nan
		bkg_smooth, bitmask = Smooth_bkg(masked, extrapolate = True)
		round1 = data - bkg_smooth
		round2 = round1 * (big_strap==1)*1
		round2[round2 == 0] = np.nan
		if np.nansum(straps) > 1:
			strap_bkg = Strap_bkg(round2)
		else:
			strap_bkg = np.zeros_like(data)
		frame_bkg = strap_bkg + bkg_smooth
		frame_bkg += np.nanmedian(frame_bkg * big_strap * big_mask)
	else:
		frame_bkg = np.zeros_like(data) * np.nan
	return frame_bkg


def Background(TPF,Mask,parallel):
	"""
	Calculate the background for the tpf, accounting for straps.

	Parameters
	----------
	TPF : lightkurve target pixel file
		tpf of interest

	Mask : array
		source mask

	parallel : bool
		determine if the background is calculated in parallel

	Returns
	-------
	bkg : array
		background for all frames in the tpf

	"""
	mask = deepcopy(Mask)
	# hack solution for new lightkurve
	if type(TPF.flux) != np.ndarray:
		data = TPF.flux.value
	else:
		data = TPF.flux

	bkg = np.zeros_like(data) * np.nan
	
	strap_mask = np.zeros_like(data[0])
	straps = pd.read_csv(package_directory + 'tess_straps.csv')['Column'].values + 44 - TPF.column
	# limit to only straps that are in this fov
	straps = straps[((straps > 0) & (straps < Mask.shape[1]))]
	strap_mask[:,straps] = 1
	big_strap = convolve(strap_mask,np.ones((3,3))) > 0
	big_mask = convolve((mask==0)*1,np.ones((3,3))) > 0
	flux = deepcopy(data)
	if parallel:
		num_cores = multiprocessing.cpu_count()
		bkg = Parallel(n_jobs=num_cores)(delayed(Calculate_bkg)(frame,straps,big_mask,big_strap) for frame in flux)
	else:
		for i in range(flux.shape[0]):
			bkg[i] = Calculate_bkg(flux[i],straps,big_mask,big_strap)
	return bkg

def Get_ref(data,start = None, stop = None):
	'''
	Get refernce image to use for subtraction and mask creation.
	The image is made from all images with low background light.

	Parameters
	----------
	data : array
		3x3 array of flux, axis: 0 = time; 1 = row; 2 = col

	Returns
	-------
	reference : array
		reference array from which the source mask is identified
	'''
	# hack solution for new lightkurve

	if type(data) != np.ndarray:
		data = data.value
	if (start is None) & (stop is None):
		d = data[np.nansum(data,axis=(1,2)) > 100]
		summed = np.nansum(d,axis=(1,2))
		lim = np.percentile(summed[np.isfinite(summed)],5)
		ind = np.where((summed < lim))[0]
		reference = np.nanmedian(d[ind],axis=(0))
	elif (start is not None) & (stop is None):
		start = int(start)
		reference = np.nanmedian(data[start:],axis=(0))

	elif (start is None) & (stop is not None):
		stop = int(stop)
		reference = np.nanmedian(data[:stop],axis=(0))

	else:
		start = int(start)
		stop = int(stop)
		reference = np.nanmedian(data[start:stop],axis=(0))
	return reference

def Calculate_shifts(data,mx,my,daofind):
	"""
	Calculate the offsets of sources identified by photutils from a reference

	Parameters
	----------
	data : array
		a single frame from the tpf

	mx : array
		mean row positions for the centroids from the reference image

	my : array
		mean col positions for the centroids from the reference image

	daofind : DAOStarFinder
		module to find the centroid positions

	Returns
	-------
	shifts : array
		row and col shift to match the data centroids to the reference image

	"""
	shifts = np.zeros((2,len(mx))) * np.nan
	if np.nansum(data) > 0:
		mean, med, std = sigma_clipped_stats(data, sigma=3.0)
		s = daofind(data - med)
		if type(s) != type(None):
			x = s['xcentroid']
			y = s['ycentroid']
			dist = np.zeros((len(mx),len(x)))
			dist = dist + np.sqrt((x[np.newaxis,:] - mx[:,np.newaxis])**2 + 
								  (y[np.newaxis,:] - my[:,np.newaxis])**2)
			ind = np.argmin(dist,axis=1)
			indo = np.nanmin(dist) < 1
			ind = ind[indo]
			shifts[0,indo] = x[ind] - mx[indo]
			shifts[1,indo] = y[ind] - my[indo]
	return shifts

def Centroids_DAO(Flux,Median,TPF=None,parallel = False):
	"""
	Calculate the centroid shifts of time series images.
	
	Parameters
	----------
	Flux : array 
		3x3 array of flux, axis: 0 = time; 1 = row; 2 = col

	Median : array
		median image used for the position reference

	TPF : lightkurve targetpixelfile
		tpf
	
	parallel : bool
		if True then parallel processing will be used for shift calculations

	Returns
	-------
	smooth : array
		smoothed displacement of the centroids compared to the Median
	"""
	# hack solution for new lightkurve
	if type(Flux) != np.ndarray:
		Flux = Flux.value

	m = Median.copy()
	f = deepcopy(Flux)#TPF.flux.copy()
	mean, med, std = sigma_clipped_stats(m, sigma=3.0)
	
	daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
	s = daofind(m - med)
	mx = s['xcentroid']
	my = s['ycentroid']
	
	if parallel:
		
		num_cores = multiprocessing.cpu_count()
		shifts = Parallel(n_jobs=num_cores)(
			delayed(Calculate_shifts)(frame,mx,my,daofind) for frame in f)
		shifts = np.array(shifts)
	else:
		for i in range(len(f)):
			shifts = np.zeros((len(f),2,len(mx))) * np.nan
			shifts[i,:,:] = Calculate_shifts(f[i],mx,my,daofind)


	meds = np.nanmedian(shifts,axis = 2)
	meds[~np.isfinite(meds)] = 0
	
	smooth = Smooth_motion(meds,TPF)
	nans = np.nansum(f,axis=(1,2)) ==0
	smooth[nans] = np.nan

	return smooth

def Smooth_motion(Centroids,tpf):
	"""
	Calculate the smoothed centroid shift 

	Parameters
	----------
	Centroids : array
		centroid shifts from all frames


	TPF : lightkurve targetpixelfile
		tpf

	Returns
	-------
	smoothed : array
		smoothed displacement of the centroids

	"""
	smoothed = np.zeros_like(Centroids) * np.nan
	try:
		split = np.where(np.diff(tpf.astropy_time.mjd) > 0.5)[0][0] + 1
		# ugly, but who cares
		ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
		ind1 = np.where(ind1 != 0)[0]
		ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
		ind2 = np.where(ind2 != 0)[0] + split
		smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],51,3)
		smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],51,3)

		smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],51,3)
		smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],51,3)
	except IndexError:
		smoothed[:,0] = savgol_filter(Centroids[:,0],51,3)		
		smoothed[:,1] = savgol_filter(Centroids[:,1],51,3)
	return smoothed


def Shift_images(Offset,Data,median=False):
	"""
	Shifts data by the values given in offset. Breaks horribly if data is all 0.

	Parameters
	----------
	Offset : array 
		centroid offsets relative to a reference image

	Data : array
		3x3 array of flux, axis: 0 = time; 1 = row; 2 = col

	median : bool
		if true then the shift direction will be reveresed to shift the reference

	Returns
	-------
	shifted : array
		array shifted to match the offsets given

	"""
	# hack solution for new lightkurve
	if type(Data) != np.ndarray:
		Data = Data.value

	shifted = Data.copy()
	data = Data.copy()
	data[data<0] = 0
	for i in range(len(data)):
		if np.nansum(data[i]) > 0:
			shifted[i] = shift(data[i],[-Offset[i,1],-Offset[i,0]],mode='nearest',order=3)
	return shifted


def Lightcurve(flux, aper, normalise = False):
	"""
	Calculate a light curve from a time series of images through aperature photometry.

	Parameters
	----------
	flux : array
		3x3 array of flux, axis: 0 = time; 1 = row; 2 = col

	aper : array
		mask for data to perform aperature photometry 

	normalise : bool
		normalise the light curve to the median

	Returns
	-------
	LC : array
		light curve 

	"""
	# hack solution for new lightkurve
	if type(flux) != np.ndarray:
		flux = flux.value

	aper[aper == 0] = np.nan
	LC = np.nansum(flux*aper, axis = (1,2))
	LC[LC == 0] = np.nan
	for k in range(len(LC)):
		if np.isnan(flux[k]*aper).all():
			LC[k] = np.nan
	if normalise:
		LC = LC / np.nanmedian(LC)
	return LC

def bin_data(flux,t,bin_size):
	"""
	Bin a light curve to the desired duration specified by bin_size

	Parameters
	----------
	flux : array
		light curve in counts 

	t : array
		time array

	bin_size : int
		number of bins to average over

	Returns
	-------
	lc : array
		time averaged light curve
	t[x] : array
		time averaged time 
	"""
	bin_size = int(bin_size)
	lc = []
	x = []
	for i in range(int(len(flux)/bin_size)):
		if np.isnan(flux[i*bin_size:(i*bin_size)+bin_size]).all():
			lc.append(np.nan)
			x.append(int(i*bin_size+(bin_size/2)))
		else:
			lc.append(np.nanmedian(flux[i*bin_size:(i*bin_size)+bin_size]))
			x.append(int(i*bin_size+(bin_size/2)))
	lc = np.array(lc)
	x = np.array(x)
	return lc, t[x]


def Make_lc(t,flux,aperture = None,bin_size=0,normalise=False,clip = False):
    """
    Perform aperature photometry on a time series of images

    Parameters
    ----------
    flux : array

    t : array
        time 

    aper : None, list, array
        aperature to do aperature photometry on.


    bin_size : int
        number of points to average

    normalise : bool
        if true the light curve is normalised to the median

    Returns
    -------
    lc : array 
        light curve for the pixels defined by the aperture
    """
    # hack solution for new lightkurve
    if type(flux) != np.ndarray:
        flux = flux.value

    if type(aperture) == type(None):
        aper = np.zeros_like(flux[0])
        aper[int(aper.shape[0]/2),int(aper.shape[1]/2)] = 1
        aper = convolve(aper,np.ones((3,3)))
        temp = np.zeros_like(flux[0])
    elif type(aperture) == list:
        temp = np.zeros_like(flux[0])
        temp[aperture[0],aperture[1]] = 1 
        aper = temp
    elif type(aperture) == np.ndarray:
        aper = aperture * 1.
         
    lc = Lightcurve(flux,aper,normalise = normalise)
    if clip:
        mask = ~sigma_mask(lc)
        lc[mask] = np.nan
    if bin_size > 1:
        lc, t = bin_data(lc,t,bin_size)
    lc = np.array([t,lc])
    return lc

def Plotter(t,flux):
	plt.figure()
	plt.plot(t,flux)
	plt.ylabel('Counts')
	plt.xlabel('Time MJD')
	plt.show()
	return


def Quick_reduce(tpf, aper = None, shift = True, parallel = True, 
					normalise = False, bin_size = 0, plot = True, all_output = True):
	"""
	Reduce the images from the target pixel file and make a light curve with aperture photometry.
	This background subtraction method works well on tpfs > 50x50 pixels.

	Parameters 
	----------
	tpf : lightkurve target pixel file
		tpf to act on 

	aper : None, list, array
		aperature to do photometry on

	shift : bool
		if True the flux array will be shifted to match the position of a reference

	parallel : bool
		if True parallel processing will be used for background estimation and centroid shifts 

	normalise : bool
		if True the light curve will be normalised to the median

	bin_size : int
		if > 1 then the lightcurve will be binned by that amount

	all_output : bool
		if True then the lc, flux, reference and background will be returned.

	Returns
	-------
	if all_output = True
		lc : array 
			light curve

		flux : array
			shifted images to match the reference

		ref : array
			reference array used in image matching
		
		bkg : array
			array of background flux avlues for each image
	
	else
		lc : array 
			light curve
	"""
	# make reference
	ref = Get_ref(tpf.flux)
	print('made reference')
	# make source mask
	mask = Source_mask(ref,grid=0)
	print('made source mask')
	# calculate background for each frame
	print('calculating background')
	try:
		bkg = Background(tpf,mask,parallel=parallel)
	except:
		print('Something went wrong, switching to serial')
		parallel = False
		bkg = Background(tpf,mask,parallel=False)

	if np.isnan(bkg).all():
		# check to see if the background worked
		raise ValueError('bkg all nans')
	
	if type(tpf.flux) != np.ndarray:
		flux = tpf.flux.value
	else:
		flux = tpf.flux

	flux = flux - bkg
	print('background subtracted')
	ref = Get_ref(flux)
	#return flux, bkg
	if np.isnan(flux).all():
		raise ValueError('flux all nans')

	if shift:
		print('calculating centroids')
		try:
			offset = Centroids_DAO(flux,ref,TPF=tpf,parallel=parallel)
		except:
			print('Something went wrong, switching to serial')
			parallel = False
			offset = Centroids_DAO(flux,ref,TPF=tpf,parallel=parallel)

		flux = Shift_images(offset,flux)

		print('images shifted')
	
	lc = Make_lc(tpf.astropy_time.mjd,flux,aperture=aper,bin_size=bin_size,normalise=normalise)
	print('made light curve')

	if all_output:
		return lc, flux, ref, bkg
	else:
		return lc



def Remove_stellar_variability(lc):
	"""
	Removes all long term stellar variability, while preserving flares. Input a light curve 
	with shape (2,n) and it should work!

	Parameters
	----------
	lc : array
		lightcurve with the shape of (2,n), where the first index is time and the second is 
		flux.

	Outputs
	-------
	trends : array
		the stellar trends, subtract this from your input lc
	"""
    # Make a smoothing value with a significant portion of the total 
    size = int(lc.shape[1] * 0.04)
    if size / 2 == int(size/2): size += 1
    smooth = savgol_filter(lc[1,:],5001,3)
    mask = sigma_clip(lc[1]-smooth,sigma_upper=3,sigma_lower=10,masked=True).mask
    ind = np.where(mask)[0]
    masked = lc.copy()
    # Mask out all peaks, with a lead in of 5 frames and tail of 100 to account for decay
    # todo: use findpeaks to get height estimates and change the buffers accordingly
    for i in ind:
        masked[:,i-5:i+100] = np.nan
    finite = np.isfinite(masked[1,:])
    ## Hack solution doesnt need to worry about interpolation. Assumes that stellar variability 
    ## is largely continuous over the missing data regions.
    #f1 = interp1d(lc[0,finite], lc[1,finite], kind='linear',fill_value='extrapolate')
    #interp = f1(lc[0,:])
    
    # Smooth the remaining data, assuming its effectively a continuous data set (no gaps)
    size = int(lc.shape[1] * 0.005)
    if size / 2 == int(size/2): size += 1
    smooth = savgol_filter(lc[1,finite],size,1)
    # interpolate the smoothed data over the missing time values
    f1 = interp1d(lc[0,finite], smooth, kind='linear',fill_value='extrapolate')
    trends = f1(lc[0])
    # huzzah, we now have a trend that should remove stellar variability, excluding flares.
    return trends 
