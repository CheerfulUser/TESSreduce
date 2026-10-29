import os
import requests
import json
from copy import deepcopy

import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO

from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import label
from scipy.ndimage import binary_dilation
from scipy.ndimage import laplace
from scipy.ndimage import median_filter
from scipy.ndimage import distance_transform_edt
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from skimage.restoration import inpaint


from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
import lightkurve as lk
from photutils.detection import StarFinder

from PRF import TESS_PRF
import tesswcs
from tabulate import tabulate

package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

from .psf_photom import create_psf



fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def strip_units(data):
	"""
	Removes the units off of data that was not in a NDarray, such as an astropy table. Returns an NDarray that has no units 

	Parameters:
	----------
	data: ArrayLike
			ArrayLike set of data that may have associated units that want to be removed. Should be able to return something sensible when .values is called.

	Returns:
	-------
	data: ArrayLike
			Same shape as input data, but will not have any units
	"""
	
	if type(data) != np.ndarray:
		data = data.value
	return deepcopy(data)

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
	# catch for if there are no pixels that escape the mask
	if np.nansum(np.isfinite(data)) > 10:
		if grid > 0:
			data[data<0] = 0
			data[data >= np.nanpercentile(data,95)] = np.nan
			grid = np.zeros_like(data)
			size = grid
			for i in range(grid.shape[0]//size):
				for j in range(grid.shape[1]//size):
					section = data[i*size:(i+1)*size,j*size:(j+1)*size]
					section = section[np.isfinite(section)]
					lim = np.nanpercentile(section,1)
					grid[i*size:(i+1)*size,j*size:(j+1)*size] = lim
			thing = data - grid
		else:
			thing = data
		ind = np.isfinite(thing)
		mask = ((thing <= np.percentile(thing[ind],95,axis=0)) &
			   (thing <= np.percentile(thing[ind],10))) * 1.
	else:
		mask = np.zeros_like(data)

	return mask

def unknown_mask(image):
	mask = np.zeros_like(image)
	for i in range(image.shape[1]):
		d = image.copy()
		m = np.array([])
		masked = image.copy()
		x = np.arange(image.shape[0])
		y = d * 1.

		y[y==0] = np.nan
		g = np.gradient(y)

		m = np.append(m,sigma_clip(g,sigma=3).mask)

		masked[m>0] = np.nan
		for k in range(5):
			nonan = np.isfinite(masked)
			filled = interp1d(x[nonan],masked[nonan],bounds_error=False,fill_value='extrapolate',kind='linear')
			filled = filled(x)
			
			sav = savgol_filter(filled,image.shape[1]//2+1,2)
			dif = masked-sav
			m2 = sigma_clip(dif,sigma=3).mask
			mm = np.zeros(len(masked))
			mm[nonan] = 1
			mask[:,i][m2] = 1
	return mask


def _shift_one(frame, s):
	if np.nansum(abs(frame)) > 0:
		return shift(frame, [s[0], s[1]], mode='nearest', order=5)
	return frame

def _shift_ref_one(ref, frame_for_check, s):
	if np.nansum(abs(frame_for_check)) > 0:
		return shift(ref, [-s[1], -s[0]], mode='nearest', order=5)
	return frame_for_check

def parallel_bkg3(data,mask):
	data = deepcopy(data)
	data[mask] = np.nan
	estimate = inpaint.inpaint_biharmonic(data,mask)
	return estimate

def _background2d_frame(frame, box_size, filter_size, sigma, maxiters, mask=None):
	from photutils.background import Background2D, MedianBackground
	from astropy.stats import SigmaClip
	sc = SigmaClip(sigma=sigma, maxiters=maxiters)
	try:
		return Background2D(frame, box_size=box_size, filter_size=filter_size,
							sigma_clip=sc, bkg_estimator=MedianBackground(),
							mask=mask, fill_value=0.0,
							exclude_percentile=50).background
	except Exception:
		return np.full_like(frame, np.nanmedian(frame))

def parallel_background2d(cube, box_size=5, filter_size=3, sigma=3, maxiters=5, n_jobs=-1, mask=None, backend='loky', verbose=0):
	masks = mask if np.ndim(mask) == 3 else [mask] * len(cube)   # 3D mask: one per frame
	return np.array(Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_background2d_frame)(frame, box_size, filter_size, sigma, maxiters, mk)
		for frame, mk in zip(cube, masks)))

def Smooth_bkg(data, gauss_smooth=0, interpolate=False, extrapolate=True):
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
	#data[data == 0] = np.nan
	if (~np.isnan(data)).any():
		x = np.arange(0, data.shape[1])
		y = np.arange(0, data.shape[0])
		arr = np.ma.masked_invalid(deepcopy(data))
		xx, yy = np.meshgrid(x, y)
		#get only the valid values
		x1 = xx[~arr.mask]
		y1 = yy[~arr.mask]
		newarr = arr[~arr.mask]
		if (len(x1) > 10) & (len(y1) > 10):
			if interpolate:
				estimate = griddata((x1, y1), newarr.ravel(),
										  (xx, yy),method='linear')
				nearest = griddata((x1, y1), newarr.ravel(),
										  (xx, yy),method='nearest')
				if extrapolate:
					estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
				
				if gauss_smooth > 0:
					estimate = gaussian_filter(estimate,gauss_smooth)

			#estimate = median_filter(estimate,5)
			else:
				# try inpaint stuff
				mask = deepcopy(arr.mask)
				mask = mask.astype(bool)
				# end inpaint
				estimate = inpaint.inpaint_biharmonic(data,mask)
				#estimate = signal.fftconvolve(estimate,self.prf,mode='same')
				if gauss_smooth > 0:
					if (np.nanmedian(estimate) < 150) & (np.nanstd(estimate) < 3):
						gauss_smooth = gauss_smooth * 4
					estimate = gaussian_filter(estimate,gauss_smooth)
		else:
			estimate = np.zeros_like(data)
	else:
		estimate = np.zeros_like(data)

	return estimate

def Calculate_shifts(data,mx,my,finder):
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
		try:
			#s = daofind(data - med)
			s = finder.find_stars(deepcopy(data)-med)
		except:
			s = None
			print('bad frame')
		if type(s) != type(None):
			x = s['xcentroid']
			y = s['ycentroid']
			dist = np.zeros((len(mx),len(x)))
			dist = dist + np.sqrt((x[np.newaxis,:] - mx[:,np.newaxis])**2 + 
								  (y[np.newaxis,:] - my[:,np.newaxis])**2)
			ind = np.argmin(dist,axis=1)
			indo = (np.nanmin(dist) < 1)
			ind = ind[indo]
			shifts[1,indo] = mx[indo] - x[ind]
			shifts[0,indo] = my[indo] - y[ind]
		else:
			shifts[0,:] = np.nan
			shifts[1,:] = np.nan
	return shifts

def image_sub(theta, image, ref):#, eimage, eref):
	dx, dy = theta
	s = shift(image,([dx,dy]),order=5, mode='nearest')
	#translation = np.float64([[1,0,dx],[0,1, dy]])
	#s = cv2.warpAffine(image, translation, image.shape[::-1], flags=cv2.INTER_CUBIC,borderValue=0)
	diff = (ref-s)**2#/(eimage + eref)
	if image.shape[0] > 50:
		return np.nansum(diff[10:-11,10:-11])
	else:
		return np.nansum(diff[5:-6,5:-6])


### TESTING CODE
#def c_shift_image(img, shift):
#	return scipy.ndimage.shift(img, shift=shift, order=5, mode='nearest')  # cubic interpolation
#
#def cost_function(shift, ref_img, moving_img):
#	shifted = shift_image(moving_img, shift)
#	diff = ref_img - shifted
#	return np.sum(diff[2:-2,2:-2]**2)  # Sum of Squared Differences (SSD)
#
#def align_subpixel(ref_img, moving_img, initial_shift=(0,0)):
#	ref_img = ref_img.astype(np.float32)
#	moving_img = moving_img.astype(np.float32)
#
#	# Minimize SSD by shifting moving_img
#	result = minimize(cost_function, initial_shift, args=(ref_img, moving_img), method='Powell')
#
#	best_shift = result.x
#	aligned_img = c_shift_image(moving_img, best_shift)
#
#	print(f"Optimal shift (y, x): {best_shift}")
#	return aligned_img, best_shift
###


def difference_shifts(image,ref):#,eimage,eref):
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
	if np.nansum(abs(image)) > 0:
		x0= [0,0]
		bds = [(-1.5,1.5),(-1.5,1.5)]
		bds = [(-3,3),(-3,3)]
		#res = minimize(image_sub,x0,args=(image,ref,eimage,eref),method = 'Powell',bounds= bds)
		res = minimize(image_sub,x0,args=(image,ref),method = 'Powell',bounds= bds)
		s = res.x
		#a,s = align_subpixel(ref,image)
	else:
		s = np.zeros((2)) * np.nan
	if (s == np.ones((2))).any():
		s = np.zeros((2)) * np.nan
	return s

def Smooth_motion(Centroids,mjd,flux,skernel=25):
	"""
	Calculate the smoothed centroid shift

	Parameters
	----------
	Centroids : array
		centroid shifts from all frames

	mjd : array
		time of each frame, used to find orbit gaps

	flux : array
		flux cube, used to identify padded/empty frames

	Returns
	-------
	smoothed : array
		smoothed displacement of the centroids

	"""
	smoothed = np.zeros_like(Centroids) * np.nan
	# skernel = int(len(tpf.flux) * 0.01) #simple way of making the smoothing window 10% of the duration
	# skernel = skernel // 2 +1
	# #skernel = 25
	# if skernel < 25:
	# 	skernel = 25
	try:
		try:
			split = np.where(np.diff(mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(flux[split:],axis=(1,2))
			ind2 = np.where(ind2 != 0)[0] + split
			smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],skernel,3)
			smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],skernel,3)

			smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],skernel,3)
			smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],skernel,3)
		except:
			split = np.where(np.diff(mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(flux[split:],axis=(1,2))
			ind2 = np.where(ind2 != 0)[0] + split
			smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],skernel//2+1,3)
			smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],skernel//2+1,3)

			smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],skernel//2+1,3)
			smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],skernel//2+1,3)

	except IndexError:
		smoothed[:,0] = savgol_filter(Centroids[:,0],skernel,3)		
		smoothed[:,1] = savgol_filter(Centroids[:,1],skernel,3)
	return smoothed


def smooth_zp(zp,time):
	"""
	Calculate the smoothed centroid shift 

	Parameters
	----------
	zp : array
		centroid shifts from all frames


	time : lightkurve targetpixelfile
		tpf

	Returns
	-------
	smoothed : array
		smoothed displacement of the centroids

	"""
	import matplotlib.pyplot as plt

	smoothed = np.zeros_like(zp) * np.nan
	plt.figure()
	plt.plot(time,zp,'.')
	try:
		split = np.where(np.diff(time) > 0.5)[0][0] + 1
		# ugly, but who cares
		ind1 = np.isfinite(zp[:split])
		ind2 = np.isfinite(zp[split:]) + split
	
		smoothed[ind1] = savgol_filter(zp[ind1],15,3)
		smoothed[ind2] = savgol_filter(zp[ind2],15,3)

		smoothed[ind1] = savgol_filter(zp[ind1],15,3)
		smoothed[ind2] = savgol_filter(zp[ind2],15,3)
	except IndexError:
		smoothed[:] = savgol_filter(zp[:],15,3)
		smoothed[:] = savgol_filter(zp[:],15,3)
	err = np.nanstd(zp - smoothed)

	return smoothed, err

def grads_rad(flux):
	"""
	Calculates the radius of the flux from the gradient of the flux, and the double gradient of the flux.  

	Parameters:
	----------
	flux: ArrayLike
			An array of flux values

	Returns:
	-------
	rad: ArrayLike
			The radius of the fluxes 
	"""
	rad = np.sqrt(np.gradient(flux)**2+np.gradient(np.gradient(flux))**2)
	return rad

def grad_flux_rad(flux):
	"""
	Calculates the radius of the flux from the gradient of the flux.  

	Parameters:
	----------
	flux: ArrayLike
			An array of flux values

	Returns:
	-------
	rad: ArrayLike
			The radius of the fluxes 
	"""
	rad = np.sqrt(flux**2+np.gradient(flux)**2)
	return rad


def _tess_pointing_table():
	"""
	Sector start/end times (MJD), sourced from tesswcs.pointings so the table
	stays current with tesswcs releases rather than a file pinned in this repo.

	Returns
	-------
	sec_times : pd.DataFrame
		Indexed by Sector, with mjd_start and mjd_end columns.
	"""
	pointings = tesswcs.pointings.to_pandas()[['Sector','Start','End']]
	pointings = pointings.rename(columns={'Start':'mjd_start','End':'mjd_end'})
	pointings['mjd_start'] = Time(pointings['mjd_start'].values,format='jd').mjd
	pointings['mjd_end'] = Time(pointings['mjd_end'].values,format='jd').mjd
	return pointings.set_index('Sector').sort_index()


def _target_sectors(ra,dec):
	"""
	Find which TESS sectors, cameras and CCDs observe a given coordinate.
	Replaces tess_stars2px_function_entry (tess-point) with tesswcs, which
	covers both archived and predicted sector pointings.

	Returns
	-------
	outSecs, outCam, outCcd, outColPix, outRowPix : np.array
		Sector number, camera, CCD, and pixel column/row for each match,
		sorted by sector.
	"""
	import logging
	coord = SkyCoord(ra,dec,unit='deg')

	level = tesswcs.log.level
	tesswcs.log.setLevel(logging.ERROR)
	secs, cams, ccds, cols, rows = [], [], [], [], []
	try:
		for sector in tesswcs.pointings['Sector']:
			sector = int(sector)
			for camera in range(1,5):
				for ccd in range(1,5):
					try:
						wcs = tesswcs.WCS.from_sector(sector,camera,ccd)
					except ValueError:
						continue
					if wcs.footprint_contains(coord):
						col, row = wcs.world_to_pixel(coord)
						secs += [sector]; cams += [camera]; ccds += [ccd]
						cols += [float(col)]; rows += [float(row)]
	finally:
		tesswcs.log.setLevel(level)

	order = np.argsort(secs)
	return (np.array(secs)[order], np.array(cams)[order], np.array(ccds)[order],
			np.array(cols)[order], np.array(rows)[order])


def sn_lookup(name,time='disc',buffer=0,print_table=True, df = False):
	"""
	Check for overlapping TESS ovservations for a transient. Uses the Open SNe Catalog for 
	discovery/max times and coordinates.

	------
	Inputs
	------
	name : str
		catalog name
	time : str or float
		reference time to use, accepted string values are disc, or max. Float times are assumed to be MJD.
	buffer : float
		overlap buffer time in days 
	
	-------
	Options
	-------
	print_table : bool 
		if true then the lookup table is printed

	-------
	Returns
	-------
	tr_list : list
		list of ra, dec, and sector that can be put into tessreduce.
	"""
	tns = True
	# try:
	# 	url = 'https://api.astrocats.space/{}'.format(name)
	# 	response = requests.get(url)
	# 	json_acceptable_string = response.content.decode("utf-8").replace("'", "").split('\n')[0]
	# 	d = json.loads(json_acceptable_string)
	# 	if list(d.keys())[0] == 'message':
	# 		#print(d['message'])

	# 		#return None
	# 		tns = True
	# 	else:
	# 		disc_t = d[name]['discoverdate'][0]['value']
	# 		disc_t = Time(disc_t.replace('/','-'))
	# 		max_t = d[name]['maxdate'][0]['value']
	# 		max_t = Time(max_t.replace('/','-'))
	# 		ra = d[name]['ra'][-1]['value']
	# 		dec = d[name]['dec'][-1]['value']
	# 		tns = False
	# except:
	# 	tns = True
	if tns:
		#print('!! Open SNe Catalog down, using TNS !!')
		if (name[:2].lower() == 'sn') | (name[:2].lower() == 'at'):
			name = name[2:]
		url = f'https://www.wis-tns.org/object/{name}'
		headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
		result = requests.get(url, headers=headers)
		if result.ok:
			ra, dec = result.text.split('<div class="alter-value">')[-1].split('<')[0].split(' ')
			ra = float(ra); dec = float(dec)
			disc_t = Time(result.text.split('<span class="name">Discovery Date</span><div class="value"><b>')[-1].split('<')[0])
			max_t = deepcopy(disc_t)

			# c = SkyCoord(ra,dec, unit=(u.hourangle, u.deg))
			# ra = c.ra.deg
			# dec = c.dec.deg

	c = SkyCoord(ra,dec, unit=(u.deg, u.deg))
	ra = c.ra.deg
	dec = c.dec.deg
	
	outSecs, outCam, outCcd, outColPix, outRowPix = _target_sectors(ra, dec)

	sec_times = _tess_pointing_table()
	if len(outSecs) > 0:
		keep = np.isin(outSecs, sec_times.index)
		outSecs, outCam, outCcd, outColPix, outRowPix = \
			outSecs[keep], outCam[keep], outCcd[keep], outColPix[keep], outRowPix[keep]

		secs = sec_times.loc[outSecs].reset_index()
		if type(time) == str:
			if (time.lower() == 'disc') | (time.lower() == 'discovery'):
				disc_start = secs['mjd_start'].values - disc_t.mjd
				disc_end = secs['mjd_end'].values - disc_t.mjd
			elif (time.lower() == 'max') | (time.lower() == 'peak'):
				disc_start = secs['mjd_start'].values - max_t.mjd
				disc_end = secs['mjd_end'].values - max_t.mjd
		else:
			disc_start = secs['mjd_start'].values - time
			disc_end = secs['mjd_end'].values - time

		covers = []
		differences = []
		tr_list = []
		tab = []
		for i in range(len(disc_start)):
			ds = disc_start[i]
			de = disc_end[i]
			if (ds-buffer <= 0) & (de + buffer >= 0):
				cover = True
				dif = 0
			elif (de+buffer < 0):
				cover = False
				dif = de
			elif (ds-buffer > 0):
				cover = False
				dif = ds
			covers += [cover]
			differences += [dif]
			tab += [[secs.Sector.values[i], cover, dif]]
			tr_list += [[ra, dec, secs.Sector.values[i], cover]]

		if print_table: 
			print(tabulate(tab, headers=['Sector', 'Covers','Time difference \n(days)'], tablefmt='orgtbl'))
		if df:
			return pd.DataFrame(tr_list, columns = ['RA', 'DEC','Sector','Covers'])
		else:
			return tr_list
	else:
		print('No TESS coverage')
		return None

def spacetime_lookup(ra,dec,time=None,buffer=0,print_table=True, df = False, print_all=False):
	"""
	Check for overlapping TESS ovservations for a transient. Uses the Open SNe Catalog for 
	discovery/max times and coordinates.

	------
	Inputs
	------
	ra : float or str
		ra of object
	dec : float or str
		dec of object
	time : float
		reference time to use, must be in MJD
	buffer : float
		overlap buffer time in days 
	
	-------
	Options
	-------
	print_table : bool 
		if true then the lookup table is printed

	-------
	Returns
	-------
	tr_list : list
		list of ra, dec, and sector that can be put into tessreduce.
	"""
	if time is None:
		print('!!! WARNING no MJD time specified, using default of 59000')
		time = 59000

	if type(ra) == str:
		c = SkyCoord(ra,dec, unit=(u.hourangle, u.deg))
		ra = c.ra.deg
		dec = c.dec.deg

	outSecs, outCam, outCcd, outColPix, outRowPix = _target_sectors(ra, dec)

	sec_times = _tess_pointing_table()
	if len(outSecs) > 0:
		keep = np.isin(outSecs, sec_times.index)
		outSecs, outCam, outCcd, outColPix, outRowPix = \
			outSecs[keep], outCam[keep], outCcd[keep], outColPix[keep], outRowPix[keep]

		secs = sec_times.loc[outSecs].reset_index()
		disc_start = secs['mjd_start'].values - time
		disc_end = secs['mjd_end'].values - time

		covers = []
		differences = []
		tr_list = []
		tab = []
		for i in range(len(disc_start)):
			ds = disc_start[i]
			de = disc_end[i]
			if (ds-buffer < 0) & (de + buffer> 0):
				cover = True
				dif = 0
			elif (de+buffer < 0):
				cover = False
				dif = de
			elif (ds-buffer > 0):
				cover = False
				dif = ds
			covers += [cover]
			differences += [dif]
			tab += [[secs.Sector.values[i], outCam[i], outCcd[i], cover, dif]]
			tr_list += [[ra, dec, secs.Sector.values[i],outCam[i], outCcd[i], cover]]
		if print_table: 
			print(tabulate(tab, headers=['Sector', 'Camera', 'CCD', 'Covers','Time difference \n(days)'], tablefmt='orgtbl'))
		if df:
			return pd.DataFrame(tr_list, columns = ['RA', 'DEC','Sector','Camera','CCD','Covers'])
		else:
			return tr_list
	else:
		print('No TESS coverage')
		return None


def par_psf_source_mask(data,prf,sigma=5):

	mean, med, std = sigma_clipped_stats(data, sigma=3.0)

	finder = StarFinder(med + sigma*std,kernel=prf,exclude_border=False)
	res = finder.find_stars(deepcopy(data))
	m = np.ones_like(data)
	if res is not None:
		x = (res['xcentroid'].value + 0.5).astype(int)
		y = (res['ycentroid'].value + 0.5).astype(int)
		fwhm = (res['fwhm'].value*1.2 + 0.5).astype(int)
		fwhm[fwhm < 6] = 6
		for i in range(len(x)):
			m[y[i]-fwhm[i]//2:y[i]+fwhm[i]//2,x[i]-fwhm[i]//2:x[i]+fwhm[i]//2] = 0
	return m


def par_psf_initialise(flux,camera,ccd,sector,column,row,cutoutSize,loc,time_ind=None,ref=False,ffi=False):
	"""
	For gathering the cutouts and PRF base.
	"""
	if time_ind is None:
		time_ind = np.arange(0,len(flux))

	if (type(loc[0]) == float) | (type(loc[0]) == np.float64) |  (type(loc[0]) == np.float32):
		loc[0] = int(loc[0]+0.5)
	if (type(loc[1]) == float) | (type(loc[1]) == np.float64) |  (type(loc[1]) == np.float32):
		loc[1] = int(loc[1]+0.5)
	if ffi:
		col = column
	else:
		col = column - int(flux.shape[2]/2-1) + loc[0] # find column and row, when specifying location on a *say* 90x90 px cutout
		row = row - int(flux.shape[1]/2-1) + loc[1] 
	try:
		prf = TESS_PRF(camera,ccd,sector,col,row) # initialise psf kernel
	except:
		return np.nan
	if ref:
		cutout = (flux+ref)[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
	else:
		if ffi:
			cutout = flux[loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
			prf = create_psf(prf,cutoutSize)
			try:
				flux,pos = par_psf_full(cutout,prf,xlim=0.5,ylim=0.5)
			except:
				flux = np.nan
			return flux
		else:
			cutout = flux[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
	prf = create_psf(prf,cutoutSize)
	return prf, cutout

def par_psf_flux(image,error,prf,shift=[0,0],bkg_poly_order=3,kernel=None):
	if np.isnan(shift)[0]:
		shift = np.array([0,0])
	prf.psf_flux(image,error,ext_shift=shift,poly_order=bkg_poly_order,kernel=kernel)
	return prf.flux, prf.eflux

def par_psf_full(cutout,error,prf,shift=[0,0],xlim=0.5,ylim=0.5):
	if np.isnan(shift)[0]:
		shift = np.array([0,0])
	prf.psf_position(cutout,error,ext_shift=shift,limx=xlim,limy=ylim)
	prf.psf_flux(cutout,error)
	pos = [prf.source_x, prf.source_y]
	return prf.flux, prf.eflux, pos


def external_save_TESS(ra,dec,sector,size=90,save_path=None,quality_bitmask='default',cache_dir=None):

	if save_path is None:
		save_path = os.getcwd()

	c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec) * u.degree, frame='icrs')
	tess = lk.search_tesscut(c,sector=sector)
	tpf = tess.download(quality_bitmask=quality_bitmask,cutout_size=size,download_dir=save_path,cache_dir=cache_dir)
	
	os.system(f'mv {tpf.path} {save_path}')
	os.system(f'rm -r {save_path}/tesscut')

	if tpf is None:
		m = 'Failure in TESScut api, not sure why.'
		raise ValueError(m)

def external_get_TESS():

	found = False
	target = None
	l = os.listdir()
	for thing in l:
		if 'astrocut.fits' in thing:
			if not found:
				target = thing
			else:

				e = 'Too Many tpfs here!'
				raise FileNotFoundError(e)
	if target is not None:
		tpf = lk.TessTargetPixelFile(target)
	else:
		e = 'No available local TPF found!'
		raise FileNotFoundError(e)
	
	return tpf

def sig_err(data,err=None,sig=5,maxiter=10):
	if sig is None:
		sig = 5
	clipped = data.copy()
	ind = np.arange(0,len(data))
	breaker = 0
	if err is not None:
		for i in range(maxiter):
			nonan = np.isfinite(clipped)
			med = np.average(clipped[nonan],weights=1/err[nonan])
			#med = np.nanmedian(clipped)
			std = np.nanstd(clipped)
			mask = (clipped-1*err > med + 3*std) #| (clipped+1*err < med - 3*std)
			clipped[mask] = np.nan
			if ~mask.any():
				break

		mask = np.isnan(clipped)
	else:
		mask = sigma_clip(data,sigma_upper=sig,sigma_lower=10).mask
	return mask	


def Identify_masks(Obj):
	"""
	Uses an iterrative process to find spacially seperated masks in the object mask.
	"""
	objsub = np.copy(Obj*1)
	Objmasks = []

	mask1 = np.zeros((Obj.shape))
	if np.nansum(objsub) > 0:
		mask1[np.where(objsub==1)[0][0]] = 1
		
		while np.nansum(objsub) > 0:
			conv = ((convolve(mask1*1,np.ones(3),mode='constant', cval=0.0)) > 0)*1.0
			objsub = objsub - mask1
			objsub[objsub < 0] = 0
			if np.nansum(conv*objsub) > 0:

				mask1 = mask1 + (conv * objsub)
				mask1 = (mask1 > 0)*1
			else:

				Objmasks.append(mask1 > 0)
				mask1 = np.zeros((Obj.shape))
				if np.nansum(objsub) > 0:
					mask1[np.where(objsub==1)[0][0]] = 1
	return np.array(Objmasks)

def auto_tail(lc,mask,err = None):
	if err is not None:
		higherr = sigma_clip(err,sigma=2).mask
	else:
		higherr = False
	masks = Identify_masks(mask*1)
	med = np.nanmedian(lc[1][~mask & ~higherr])
	std = np.nanstd(lc[1][~mask & ~higherr])

	if lc.shape[1] > 4000:
		tail_length = 50
		start_length = 10

	else:
		tail_length = 5
		start_length = 1
			
	for i in range(len(masks)):
		m = np.argmax(lc[1]*masks[i])
		sig = (lc[1][m] - med) / std
		median = np.nanmedian(sig[sig>0])
		if median > 50:
			sig = sig / 100
			#sig[(sig < 1) & (sig > 0)] = 1
		if sig > 20:
			sig = 20
		if sig < 0:
			sig = 0
		masks[i][int(m-sig*start_length):int(m+tail_length*sig)] = 1
		masks[i] = masks[i] > 0
	summed = np.nansum(masks*1,axis=0)
	mask = summed > 0 
	return ~mask
		
def Multiple_day_breaks(lc):
	"""
	If the TESS data has a section of data isolated by at least a day either side,
	it is likely poor data. Such regions are identified and removed.
	
	Inputs:
	-------
	Flux - 3d array
	Time - 1d array
	
	Output:
	-------
	removed_flux - 3d array
	"""
	ind = np.where(~np.isnan(lc[1]))[0] 
	breaks = np.array([np.where(np.diff(lc[0][ind]) > .5)[0] +1])
	breaks = np.insert(breaks,0,0)
	breaks = np.append(breaks,len(lc[0]))
	return breaks

#### CLUSTERING 

def Cluster_lc(lc):
	from sklearn.cluster import OPTICS
	arr = np.array([np.gradient(lc[1]),lc[1]])
	clust = OPTICS(min_samples=12, xi=.05, min_cluster_size=.05)
	opt = clust.fit(arr.T)
	lab = opt.labels_
	keys = np.unique(opt.labels_)
	
	m = np.zeros(len(keys))
	for i in range(len(keys)):
		m[i] = np.nanmedian(lc[1,keys[i]==lab])
	bkg_ind = lab == keys[np.nanargmin(m)]
	other_ind = ~bkg_ind
	
	return bkg_ind, other_ind


def Cluster_cut(lc,err=None,sig=3,smoothing=True,buffer=48*2):
	bkg_ind, other_ind = Cluster_lc(lc)
	leng = 5
	if smoothing:
		for i in range(leng-2):
			kern = np.zeros((leng))
			kern[[0, -1]] = 1
			other_ind[convolve(other_ind*1, kern) > 1] = True
			leng -= 1
	segments = Identify_masks(other_ind)
	clipped = lc[1].copy()
	med = np.nanmedian(clipped[bkg_ind])
	std = np.nanstd(clipped[bkg_ind])
	if err is not None:
		mask = (clipped-1*err > med + sig*std)
	else:
		mask = (clipped > med + sig*std)
	overlap = np.nansum(mask * segments,axis=1) > 0
	mask = np.nansum(segments[overlap],axis=0)>0 
	mask = convolve(mask,np.ones(buffer)) > 0
	return mask


def _Get_images(ra,dec,filters):
	
	"""Query ps1filenames.py service to get a list of images"""
	
	service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
	url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
	table = Table.read(url, format='ascii')
	return table

def _Get_url(ra, dec, size, filters, color=False):
	
	"""Get URL for images in the table"""
	
	table = _Get_images(ra,dec,filters=filters)
	url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
		   f"ra={ra}&dec={dec}&size={size}&format=jpg")
   
	# sort filters from red to blue
	flist = ["yzirg".find(x) for x in table['filter']]
	table = table[np.argsort(flist)]
	if color:
		if len(table) > 3:
			# pick 3 filters
			table = table[[0,len(table)//2,len(table)-1]]
		for i, param in enumerate(["red","green","blue"]):
			url = url + "&{}={}".format(param,table['filename'][i])
	else:
		urlbase = url + "&red="
		url = []
		for filename in table['filename']:
			url.append(urlbase+filename)
	return url

def _Get_im(ra, dec, size,color):
	
	"""Get color image at a sky position"""

	if color:
		url = _Get_url(ra,dec,size=size,filters='grz',color=True)
		r = requests.get(url)
	else:
		url = _Get_url(ra,dec,size=size,filters='i')
		r = requests.get(url[0])
	im = Image.open(BytesIO(r.content))
	return im

def _Panstarrs_phot(ra,dec,size):
	import matplotlib.pyplot as plt

	grey_im = _Get_im(ra,dec,size=size*4,color=False)
	colour_im = _Get_im(ra,dec,size=size*4,color=True)

	plt.rcParams.update({'font.size':12})
	plt.figure(1,figsize=(3*fig_width,1*fig_width))
	plt.subplot(121)
	plt.imshow(grey_im,origin="lower",cmap="gray")
	plt.title('PS1 i')
	plt.xlabel('px (0.25")')
	plt.ylabel('px (0.25")')
	plt.subplot(122)
	plt.title('PS1 grz')
	plt.imshow(colour_im,origin="lower")
	plt.xlabel('px (0.25")')
	plt.ylabel('px (0.25")')


def _Skymapper_phot(ra,dec,size):
	"""
	Gets g,r,i from skymapper.
	"""
	import matplotlib.pyplot as plt

	size /= 3600

	url = f"https://api.skymapper.nci.org.au/public/siap/dr2/query?POS={ra},{dec}&SIZE={size}&BAND=g,r,i&FORMAT=GRAPHIC&VERB=3"
	table = Table.read(url, format='ascii')

	# sort filters from red to blue
	flist = ["irg".find(x) for x in table['col3']]
	table = table[np.argsort(flist)]

	if len(table) > 3:
		# pick 3 filters
		table = table[[0,len(table)//2,len(table)-1]]

	plt.rcParams.update({'font.size':12})
	plt.figure(1,figsize=(3*fig_width,1*fig_width))

	plt.subplot(131)
	url = table[2][3]
	r = requests.get(url)
	im = Image.open(BytesIO(r.content))
	plt.imshow(im,origin="upper",cmap="gray")
	plt.title('SkyMapper g')
	plt.xlabel('px (1.1")')

	plt.subplot(132)
	url = table[1][3]
	r = requests.get(url)
	im = Image.open(BytesIO(r.content))
	plt.title('SkyMapper r')
	plt.imshow(im,origin="upper",cmap="gray")
	plt.xlabel('px (1.1")')

	plt.subplot(133)
	url = table[0][3]
	r = requests.get(url)
	im = Image.open(BytesIO(r.content))
	plt.title('SkyMapper i')
	plt.imshow(im,origin="upper",cmap="gray")
	plt.xlabel('px (1.1")')

def event_cutout(coords,size=50,phot=None):

	if phot is None:
		if coords[1] > -10:
			phot = 'PS1'
		else:
			phot = 'SkyMapper'
		
	if phot == 'PS1':
		_Panstarrs_phot(coords[0],coords[1],size)

	elif phot.lower() == 'skymapper':
		_Skymapper_phot(coords[0],coords[1],size)

	else:
		print('Photometry name invalid.')

def Extract_fits(pixelfile):
	"""
	Quickly extract fits
	"""
	try:
		hdu = fits.open(pixelfile)
		return hdu
	except OSError:
		print('OSError ',pixelfile)
		return

def _clip_region(image, rx, ry, sigma, iters):
	m, me, s = sigma_clipped_stats(image[ry, rx], maxiters=iters)
	cut = (image[ry, rx] >= me + sigma * s) | (image[ry, rx] <= me - sigma * s)
	return rx[cut], ry[cut]

def regional_stats_mask(image, size=90, sigma=3, iters=10, n_jobs=1, backend='loky', verbose=0):
	if size < 30:
		print('!!! Region size is small !!!')
	sx, sy = image.shape
	X, Y = np.ogrid[0:sx, 0:sy]
	regions = sy // size * (X // size) + Y // size
	max_reg = int(np.max(regions))

	region_pixels = [np.where(regions == i) for i in range(max_reg + 1)]

	clip = np.zeros_like(image)
	results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_clip_region)(image, rx, ry, sigma, iters)
		for rx, ry in region_pixels)
	for rx_cut, ry_cut in results:
		clip[rx_cut, ry_cut] = 1
	return clip



def subdivide_region(flux,ideal_size=90):
	sx, sy = flux.shape
	valid = np.arange(0,101)
	ystep = valid[np.where(sy / valid == sy // valid)[0]]
	xstep = valid[np.where(sx / valid == sx // valid)[0]]

	ysteps = ystep[np.argmin((abs((sy / ystep) - ideal_size)))].astype(int)
	xsteps = xstep[np.argmin((abs((sx / xstep) - ideal_size)))].astype(int)
	ystep = sy//ystep[np.argmin((abs((sy / ystep) - ideal_size)))].astype(int)
	xstep = sx//xstep[np.argmin((abs((sx / xstep) - ideal_size)))].astype(int)

	regions = np.zeros_like(flux)
	counter = 0
	for i in range(ysteps):
		for j in range(xsteps):
			regions[i*ystep:(i+1)*ystep,j*xstep:(j+1)*xstep] = counter
			counter += 1
			
	max_reg = np.max(regions)
	return regions, max_reg, ystep, xstep

def Surface_names2model(names):
	# C[i] * X^n * Y^m
	return ' + '.join([
				f"C[{i}]*{n.replace(' ','*')}"
				for i,n in enumerate(names)])

def clip_background(bkg,mask,sigma=3,kern_size=5):
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.linear_model import LinearRegression
	from sklearn.pipeline import make_pipeline
	regions, max_reg, ystep, xstep = subdivide_region(bkg)
	b2 = deepcopy(bkg)
	for j in range(2):
		for region in range(int(max_reg)):
			ry,rx = np.where(regions == region)
			y = ry.reshape(ystep,xstep)
			x = rx.reshape(ystep,xstep)
			sm = abs((mask & 1)-1)[y,x] * 1.0
			sm[sm==0] = np.nan
			cut = b2[y,x]
			if j > 0:
				masked = cut * sm
			else:
				masked = cut
			xx = x.reshape(-1,1)
			yy = y.reshape(-1,1)
			zz = masked.reshape(-1,1)
			ind = np.where(np.isfinite(zz))
			order = 6
			model = make_pipeline(PolynomialFeatures(degree=order),
										 LinearRegression(fit_intercept=False))
			model.fit(np.c_[xx[ind], yy[ind]], zz[ind])
			m = Surface_names2model(model[0].get_feature_names_out(['X', 'Y']))
			C = model[1].coef_.T  # coefficients
			r2 = model.score(np.c_[xx[ind], yy[ind]], zz[ind])  # R-squared
			ZZ = model.predict(np.c_[x.flatten(), y.flatten()]).reshape(x.shape)
			diff = cut - ZZ
			m,me, s = sigma_clipped_stats(diff,maxiters=10)
			ind_arr = (diff >= (me+sigma*s)) | (diff <= (me-sigma*s))
			ind_arr = fftconvolve(ind_arr,np.ones((kern_size,kern_size)),mode='same')
			ind_arr = ind_arr > 0.8
			cut_ind = np.where(ind_arr)
			bkg2 = deepcopy(cut)
			bkg2[cut_ind] = ZZ[cut_ind]
			#bkg2 = ZZ
			b2[y,x] = bkg2
	return b2

def grad_clip_fill_bkg(bkg,sigma=3,max_size=1000):
	a,b = np.gradient(bkg)
	c = np.nanmax([abs(a),abs(b)],axis=0)
	a_mean,a_med,a_std = sigma_clipped_stats(abs(a),maxiters=10)
	b_mean,b_med,b_std = sigma_clipped_stats(abs(b),maxiters=10)
	bp = (abs(b) - b_med) > 3*b_std
	bp = fftconvolve(bp,np.ones((3,3)),mode='same') > 0.8
	ap = (abs(a) - a_med) > 3*a_std
	ap = fftconvolve(ap,np.ones((3,3)),mode='same') > 0.8

	b_labeled, b_objects = label(bp)
	a_labeled, a_objects = label(ap)

	# Component sizes via bincount — labels 1..n_objects, index 0 is background
	b_obj_size = np.bincount(b_labeled.ravel(), minlength=b_objects + 1)
	a_obj_size = np.bincount(a_labeled.ravel(), minlength=a_objects + 1)

	# Zero out components outside the valid size range (skip label 0 = background)
	bad_a = np.where((a_obj_size[1:] >= max_size) | (a_obj_size[1:] <= 9))[0] + 1
	if len(bad_a) > 0:
		a_labeled[np.isin(a_labeled, bad_a)] = 0

	bad_b = np.where((b_obj_size[1:] >= max_size) | (b_obj_size[1:] <= 9))[0] + 1
	if len(bad_b) > 0:
		b_labeled[np.isin(b_labeled, bad_b)] = 0

	overlap = (a_labeled > 0) & (b_labeled > 0)
	y, x = np.where(overlap)

	good_a = np.unique(a_labeled[y, x])
	good_b = np.unique(b_labeled[y, x])

	# Overlap ratios via bincount — counts per label, then fraction overlapping
	_a_max = int(a_labeled.max()) + 1 if a_labeled.max() > 0 else 1
	_b_max = int(b_labeled.max()) + 1 if b_labeled.max() > 0 else 1
	a_lab_flat = a_labeled.ravel()
	b_lab_flat = b_labeled.ravel()
	ov_flat = overlap.ravel().astype(float)

	a_sizes = np.bincount(a_lab_flat, minlength=_a_max)
	a_overlap_counts = np.bincount(a_lab_flat, weights=ov_flat, minlength=_a_max)
	a_ratio = a_overlap_counts[good_a] / np.maximum(a_sizes[good_a], 1)

	b_sizes = np.bincount(b_lab_flat, minlength=_b_max)
	b_overlap_counts = np.bincount(b_lab_flat, weights=ov_flat, minlength=_b_max)
	b_ratio = b_overlap_counts[good_b] / np.maximum(b_sizes[good_b], 1)

	suppress_a = good_a[a_ratio < 0.2]
	if len(suppress_a) > 0:
		a_labeled[np.isin(a_labeled, suppress_a)] = 0

	suppress_b = good_b[b_ratio < 0.2]
	if len(suppress_b) > 0:
		b_labeled[np.isin(b_labeled, suppress_b)] = 0

	c = (a_labeled + b_labeled) > 0

	c_labeled, c_objects = label(c == 0)
	if c_objects > 0:
		c_sizes = np.bincount(c_labeled.ravel(), minlength=c_objects + 1)[:c_objects]
		small = np.where(c_sizes < 10)[0]
		if len(small) > 0:
			c[np.isin(c_labeled, small)] = 1
	
	#points = fftconvolve(c,np.ones((5,5)),mode='same')
	points = c>0#oints > 0.8
	data = deepcopy(bkg)
	data[points] = np.nan
	estimate = inpaint.inpaint_biharmonic(data,points)
	return estimate

def grad_clip(data,box_size=100):
	"""
	Perform a local sigma clip of points based on the gradient of the points. 
	Pixels with large gradients are contaminated by stars/galaxies.

	Inputs
	------
		data : array
			1d array of the data to clip
		box_size : int 
			integer defining the box size to clip over 
	Output
	------
		gradind : bool

	"""
	gradind = np.zeros_like(data)
	
	for i in range(len(data)):
		if i < box_size//2:
			d = data[:i+box_size//2]
		elif len(data) - i < box_size//2:
			d = data[i-box_size//2:]
		else:
			d = data[i-box_size//2:i+box_size//2]
		
		ind = np.isfinite(d)
		d = d[ind]
		if len(d) > 5:
			gind = ~sigma_clip(np.gradient(abs(d))+d,sigma_lower=5,).mask
			if i < box_size//2:
				gradind[:i+box_size//2][ind] = gind
			elif len(data) - i < box_size//2:
				gradind[i-box_size//2:][ind] = gind
			else:
				gradind[i-box_size//2:i+box_size//2][ind] = gind
	
	gradind = gradind > 0
	return gradind 

def fit_strap(data,mask):
	"""
	interpolate over missing data

	"""
	x = np.arange(0,len(data))
	y = data.copy()
	p = np.ones_like(x)
	if len(y[mask]) > 5:
		p = interp1d(x[mask], y[mask],bounds_error=False,fill_value=np.nan,kind='linear')
		p = p(x)
		if np.isnan(p).any():
			p2 = interp1d(x[mask], y[mask],bounds_error=False,fill_value='extrapolate',kind='nearest')
			p2 = p2(x)
			p[np.isnan(p)] = p2[np.isnan(p)]
	return p

def _strap_fit_col(col_data, norm_col):
	d = abs(np.gradient(col_data))
	m, med, std = sigma_clipped_stats(d, maxiters=10)
	nm = (d > med + std) * 1
	nm = np.convolve(nm, np.ones(3), mode='same')
	nm = (nm == 0)
	q = fit_strap(norm_col, nm)
	if len(col_data) > 110:
		q = savgol_filter(q, 101, 1)
	else:
		q[:] = np.nanmedian(q)
	return q

def parallel_strap_fit(frame, frame_bkg, frame_err, mask, repeats=3, tol=3, n_jobs=1, backend='loky', verbose=0):
	norm = frame / frame_bkg
	sind = np.where(np.nansum(mask, axis=0) > 0)[0]
	qe = np.ones_like(frame)
	if len(sind) == 0:
		return qe
	results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_strap_fit_col)(frame[:, i], norm[:, i])
		for i in sind)
	for col, q in zip(sind, results):
		qe[:, col] = q
	return qe


def simulate_epsf(camera,ccd,sector,column,row,size=13,realizations=2000,oversample=3):
	from photutils.psf import EPSFBuilder, EPSFStar, EPSFStars

	psfs = []
	randpos = np.random.uniform(-1,1,size=(2,realizations)) + 6
	prf_mod = TESS_PRF(cam=camera, ccd=ccd,sector=sector,
						colnum=column,rownum=row)

	for i in range(len(randpos)):
		psfs += [prf_mod.locate(randpos[0,i],randpos[1,i],(13,13))]
	psfs = np.array(psfs)

	stars = []
	for i in range(len(psfs)):
		stars += [EPSFStar(psfs[i],cutout_center=randpos[:,i])]
	stars = EPSFStars(stars)

	epsf_builder = EPSFBuilder(oversampling=oversample, maxiters=20, progress_bar=False)
	epsf, fitted_stars = epsf_builder(stars)

	return epsf

def parallel_photutils(cutout,e_cutout,psf_phot,init_params=None,return_pos=False):
	if np.nansum(abs(cutout)) > 0:
		phot = psf_phot(cutout, error=e_cutout,init_params=init_params)
		phot = phot.to_pandas()
		f = phot[['flux_fit']].values[0]; ef = phot[['flux_err']].values[0]
		pos = phot[['x_fit','y_fit']].values
		epos = phot[['x_err','y_err']].values
		if return_pos:
			return f, ef, pos, epos
		else:
			return f, ef
		#return float(phot[['flux_fit']].values), float(phot[['flux_err']].values)
	else:
		if return_pos:
			pos = np.array([np.nan,np.nan])
			return np.array([np.nan]), np.array([np.nan]), pos, pos
		else:
			return np.array([np.nan]), np.array([np.nan])


def _fix_bkg_frame(bkg_i, flux_i, bkgmask_i, prev_i, high_bkg_i,
				   strap, strap_cols, has_straps, src_mask_2d,
				   eff_box, disk, yy, xx, NY, NX,
				   n_sigma, anom_box, anom_box_fine, sep_thresh, sep_snr_thresh,
				   sep_validate, gauss_smooth, bad_bkg_sigma, bad_bkg_min_area):
	from photutils.background import Background2D, MedianBackground
	data_src = np.isnan(bkgmask_i) if bkgmask_i is not None else src_mask_2d
	phot_mask = strap | data_src
	frame = bkg_i.copy()
	excess = np.zeros(len(strap_cols))
	if has_straps and flux_i is not None:
		_, excess, _ = sigma_clipped_stats((flux_i - frame)[:, strap_cols], axis=0)
		frame[:, strap_cols] -= excess
	try:
		bkg2d = Background2D(frame, box_size=eff_box, filter_size=3,
							 mask=phot_mask, bkg_estimator=MedianBackground(),
							 exclude_percentile=50)
		trend = bkg2d.background
	except Exception:
		trend = np.full_like(frame, np.nanmedian(frame))
	resid = frame - trend
	valid_mask = ~phot_mask

	def _block_sigma(r, box):
		sigma = np.full((NY, NX), np.inf, dtype=float)
		row_starts = [min(r0, NY - box) for r0 in range(0, NY, box)]
		col_starts = [min(c0, NX - box) for c0 in range(0, NX, box)]
		for r0 in row_starts:
			for c0 in col_starts:
				r1, c1 = r0 + box, c0 + box
				vals = r[r0:r1, c0:c1][valid_mask[r0:r1, c0:c1]]
				if vals.size >= 4:
					m = np.nanmedian(vals)
					sigma[r0:r1, c0:c1] = 1.4826 * np.nanmedian(np.abs(vals - m))
		return sigma

	is_high_bkg = bool(high_bkg_i)
	sharp_mask = np.zeros((NY, NX), dtype=bool)
	if not is_high_bkg:
		sigma_coarse = _block_sigma(resid, anom_box)
		flagged_coarse = (np.abs(resid) > n_sigma * sigma_coarse) & ~phot_mask
		import sep as _sep
		lap_abs = np.abs(laplace(frame)).astype(np.float64)
		_bkg = _sep.Background(lap_abs)
		lap_sub = lap_abs - _bkg.back()
		lap_err = _bkg.rms()
		snr_map = lap_sub / (lap_err + 1e-10)
		try:
			objects = _sep.extract(lap_sub, thresh=sep_thresh, err=lap_err)
		except Exception:
			objects = []
		sep_mask = np.zeros((NY, NX), dtype=bool)
		noise = np.nanmedian(lap_err)
		for obj in objects:
			ap_mask = np.zeros((NY, NX), dtype=bool)
			_sep.mask_ellipse(ap_mask, obj['x'], obj['y'], obj['a'], obj['b'], obj['theta'], r=3.0)
			if ap_mask.sum() == 0 or snr_map[ap_mask].mean() <= sep_snr_thresh:
				continue
			cx, cy = obj['x'], obj['y']
			dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
			true_r = None
			for r in range(2, 20):
				ann = (dist >= r - 0.5) & (dist < r + 0.5)
				if ann.sum() == 0:
					break
				if lap_sub[ann].mean() < noise:
					true_r = r - 1
					break
			else:
				true_r = 19
			if true_r is None or true_r < 2 or true_r > 5:
				continue
			sep_mask |= dist <= true_r
		lap_med = np.nanmedian(lap_abs)
		lap_mad = np.nanmedian(np.abs(lap_abs - lap_med))
		is_sharp = lap_abs > lap_med + 3 * 1.4826 * lap_mad
		sharp_mask |= sep_mask
		if sep_validate:
			edge_border = np.zeros((NY, NX), dtype=bool)
			edge_border[0, :] = True; edge_border[-1, :] = True
			edge_border[:, 0] = True; edge_border[:, -1] = True
			labeled, n_comp = label(flagged_coarse)
			if n_comp > 0:
				lab_flat = labeled.ravel()
				touches_edge = np.zeros(n_comp + 1, dtype=bool)
				touches_sep_lbl = np.zeros(n_comp + 1, dtype=bool)
				touches_sharp_lbl = np.zeros(n_comp + 1, dtype=bool)
				np.bitwise_or.at(touches_edge, lab_flat, edge_border.ravel())
				np.bitwise_or.at(touches_sep_lbl, lab_flat, sep_mask.ravel())
				np.bitwise_or.at(touches_sharp_lbl, lab_flat, is_sharp.ravel())
				is_sharp_lbl = ~(touches_edge & ~touches_sep_lbl) & touches_sharp_lbl & touches_sep_lbl
				is_sharp_lbl[0] = False
				sharp_mask |= is_sharp_lbl[labeled]
		else:
			sharp_mask |= flagged_coarse & is_sharp
		smooth_mask = flagged_coarse & ~sharp_mask
		any_anom = sharp_mask.any() or smooth_mask.any()
		if any_anom:
			eff_fine = max(min(anom_box_fine, min(NY, NX) // 2), 4)
			try:
				bkg2d_fine = Background2D(frame, box_size=eff_fine, filter_size=3,
										 mask=phot_mask | sharp_mask, bkg_estimator=MedianBackground(),
										 exclude_percentile=50)
				trend_fine = bkg2d_fine.background
			except Exception:
				trend_fine = trend
			if sharp_mask.any():
				frame[sharp_mask & ~phot_mask] = trend_fine[sharp_mask & ~phot_mask]
			resid_fine = frame - trend_fine
			sigma_fine = _block_sigma(resid_fine, anom_box_fine)
			flagged_fine = (np.abs(resid_fine) > n_sigma * sigma_fine) & ~phot_mask
			confirmed = smooth_mask & flagged_fine
			if confirmed.any():
				dilated = binary_dilation(confirmed, structure=disk) & ~phot_mask
				frame[dilated] = trend_fine[dilated]
	if prev_i is not None and flux_i is not None:
		resid_new = np.abs(flux_i - frame)
		resid_prev = np.abs(flux_i - prev_i)
		delta = resid_new - resid_prev
		_, _, scale = sigma_clipped_stats(resid_prev)
		w = np.clip(delta / (scale + 1e-10), 0, 1)
		w_zero = (w == 0)
		if w_zero.any():
			w_med = median_filter(w_zero.astype(float), size=7)
			diff_mask = w_zero & ~(w_med > 0.5)
			w[w_med > 0.5] = 0
			if diff_mask.any():
				w[diff_mask] = np.nan
				_, idx = distance_transform_edt(np.isnan(w), return_indices=True)
				w[diff_mask] = w[tuple(idx[:, diff_mask])]
		w[sharp_mask] = 0.0
		w[phot_mask] = 0.0
		frame = (1 - w) * frame + w * prev_i
	smooth_sigma = 2.0 if is_high_bkg else gauss_smooth
	fixed = gaussian_filter(frame, sigma=smooth_sigma)
	if has_straps and flux_i is not None:
		_, excess, _ = sigma_clipped_stats((flux_i - fixed)[:, strap_cols], axis=0)
		fixed[:, strap_cols] += excess
	lap_abs = np.abs(laplace(fixed))
	lap_med = np.nanmedian(lap_abs)
	lap_mad = np.nanmedian(np.abs(lap_abs - lap_med))
	high_lap = lap_abs > lap_med + bad_bkg_sigma * 1.4826 * lap_mad
	labeled_lap, n_lap = label(high_lap)
	if n_lap > 0:
		areas = np.bincount(labeled_lap.ravel(), minlength=n_lap + 1)
		large = np.flatnonzero(areas[1:] >= bad_bkg_min_area) + 1
		bad_bkg_mask = np.isin(labeled_lap, large) if len(large) > 0 else np.zeros((NY, NX), dtype=bool)
	else:
		bad_bkg_mask = np.zeros((NY, NX), dtype=bool)
	return fixed, excess, sharp_mask, bad_bkg_mask


def _fit_residual_bkg(residual, exclude_mask, res_box, n_sigma):
	from photutils.background import Background2D, MedianBackground
	from astropy.stats import SigmaClip
	if (~exclude_mask).sum() < 4:
		return np.zeros_like(residual)
	sc = SigmaClip(sigma=3.0, maxiters=5)
	finite_vals = residual[~exclude_mask & np.isfinite(residual)]
	med = np.nanmedian(finite_vals) if finite_vals.size > 0 else 0.0
	std = np.nanstd(finite_vals) if finite_vals.size > 0 else 0.0
	transient_mask = exclude_mask | (np.abs(residual - med) > 5 * std)
	try:
		b = Background2D(residual, box_size=res_box, filter_size=3,
						 sigma_clip=sc, bkg_estimator=MedianBackground(),
						 mask=transient_mask, fill_value=0.0)
		corr = b.background
	except Exception:
		valid = residual[~transient_mask]
		corr = np.full_like(residual, np.nanmedian(valid) if valid.size > 0 else 0.0)
	corr_resid = residual - corr
	_, corr_med, corr_std = sigma_clipped_stats(corr_resid[~exclude_mask])
	flagged = np.abs(corr_resid) > n_sigma * corr_std
	if flagged.any():
		lap_abs = np.abs(laplace(corr_resid))
		lap_med = np.nanmedian(lap_abs)
		lap_mad = np.nanmedian(np.abs(lap_abs - lap_med))
		is_sharp = lap_abs > lap_med + 3 * 1.4826 * lap_mad
		labeled_c, n_c = label(flagged)
		if n_c > 0:
			lab_flat = labeled_c.ravel()
			comp_sizes = np.bincount(lab_flat, minlength=n_c + 1)
			sharp_counts = np.bincount(lab_flat, weights=is_sharp.ravel(), minlength=n_c + 1)
			sharp_frac = sharp_counts / np.maximum(comp_sizes, 1)
			suppress = np.flatnonzero(sharp_frac[1:] >= 0.3) + 1
			if len(suppress) > 0:
				corr[np.isin(labeled_c, suppress)] = 0.0
	return corr


def fix_background_anomalies(bkg, mask, flux=None, bkg_prev=None, bkgmask=None, n_sigma=5.0,
							  box_size=16, anom_box=30, anom_box_fine=4, dilate_r=2, gauss_smooth=2,
							  sep_thresh=3.0, sep_snr_thresh=2.0, sep_validate=True,
							  high_bkg_frames=None, high_bkg_thresh=200.0,
							  bad_bkg_sigma=10.0, bad_bkg_min_area=100, n_jobs=-1, backend='loky', verbose=0):
	"""
	Fix anomalies (asteroids, cosmic rays) in a background cube.

	Pipeline:
	  Per frame (parallel):
	    1. Subtract per-column additive excess on strap columns (bit 4).
	    2. Fit a large-scale spatial trend with photutils Background2D.
	    3. Flag pixels where |residual| > n_sigma * robust_sigma.
	    4. Replace flagged (+ dilated) pixels with the trend value.
	    5. Gaussian-smooth the result (strap excess held back).
	  Then (if flux and bkgmask supplied):
	    6. Fit and subtract a residual surface from (flux - bkg).
	  Finally:
	    7. Restore the strap column excess.

	Parameters
	----------
	bkg : ndarray (T, NY, NX)
	mask : ndarray (NY, NX) or (T, NY, NX), bitmask — bit 1 = source, bit 4 = strap
	flux : ndarray (T, NY, NX), optional — raw flux cube for residual surface step
	bkgmask : ndarray (NY, NX) or (T, NY, NX), optional — NaN-where-excluded mask
	n_sigma : float
	box_size : int
	dilate_r : int
	gauss_smooth : float
	n_jobs : int  (passed to joblib; -1 = all cores)

	Returns
	-------
	bkg_fixed : ndarray (T, NY, NX)
	"""
	from photutils.background import Background2D, MedianBackground

	T, NY, NX = bkg.shape
	mask2d = mask[0] if mask.ndim == 3 else mask
	if high_bkg_frames is None:
		src_mask = ((mask2d & 1) == 0) & ((mask2d & 2) == 0)
		ref = flux if flux is not None else bkg
		masked_ref = ref * src_mask[np.newaxis].astype(float)
		masked_ref[masked_ref == 0] = np.nan
		frame_med = np.nanmedian(masked_ref, axis=(1, 2))
		high_bkg_frames = frame_med > high_bkg_thresh
	else:
		high_bkg_frames = np.asarray(high_bkg_frames, dtype=bool)
	strap = (mask2d & 4).astype(bool)
	good_cols = np.where(~strap.any(axis=0))[0]
	strap_cols = np.where(strap.any(axis=0))[0]
	has_straps = len(strap_cols) > 0 and len(good_cols) > 0
	bkgmask_arr = np.asarray(bkgmask) if bkgmask is not None else None
	src_mask_2d = (mask2d & 1).astype(bool)

	# Scale box_size to the cutout — must fit at least 2 boxes per axis
	eff_box = min(box_size, min(NY, NX) // 2)
	eff_box = max(eff_box, 4)

	yr, xr = np.ogrid[-dilate_r:dilate_r+1, -dilate_r:dilate_r+1]
	disk = xr**2 + yr**2 <= dilate_r**2
	yy, xx = np.mgrid[:NY, :NX]

	def _bkgmask_i(i):
		if bkgmask_arr is None:
			return None
		return bkgmask_arr[i] if bkgmask_arr.ndim == 3 else bkgmask_arr

	results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_fix_bkg_frame)(
			bkg[i],
			flux[i] if flux is not None else None,
			_bkgmask_i(i),
			np.asarray(bkg_prev)[i] if bkg_prev is not None else None,
			bool(high_bkg_frames[i]) if high_bkg_frames is not None else False,
			strap, strap_cols, has_straps, src_mask_2d,
			eff_box, disk, yy, xx, NY, NX,
			n_sigma, anom_box, anom_box_fine, sep_thresh, sep_snr_thresh,
			sep_validate, gauss_smooth, bad_bkg_sigma, bad_bkg_min_area,
		)
		for i in range(T)
	)
	bkg_fixed = np.array([r[0] for r in results])
	excesses = [r[1] for r in results]
	sharp_masks = np.array([r[2] for r in results])
	bad_bkg_masks = np.array([r[3] for r in results])  # (T, NY, NX) bool

	if flux is not None and bkgmask is not None:
		res_box = min(20, min(NY, NX) // 2)
		res_box = max(res_box, 4)

		residuals = flux - bkg_fixed
		corrections = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
			delayed(_fit_residual_bkg)(
				residuals[i],
				np.isnan(bkgmask_arr[i]) if bkgmask_arr.ndim == 3 else np.isnan(bkgmask_arr),
				res_box, n_sigma,
			) for i in range(T)
		)
		bkg_fixed += np.array(corrections)

	# if has_straps:
		# for i, excess in enumerate(excesses):
			# use view to avoid numpy advanced-indexing shape ambiguity
			# frame_view = bkg_fixed[i]
			# frame_view[:, strap_cols] += excess

	return bkg_fixed, sharp_masks, bad_bkg_masks


def get_orbit_segments(times_mjd, sector=None, camera=None, vector_path=None,
					   gap_threshold=0.5):
	"""
	Assign each frame to an orbit segment.

	Tries to use TESSVectors (FFI cadence) to get official Segment assignments.
	Falls back to splitting on time gaps > gap_threshold days if unavailable.

	Returns
	-------
	segments : ndarray (T,) of int  — 1-indexed orbit labels
	"""
	import pandas as pd

	if sector is not None and camera is not None:
		fname = f'TessVectors_S{sector:03d}_C{camera}_FFI.csv'
		url = ('https://heasarc.gsfc.nasa.gov/docs/tess/data/TESSVectors/'
				 f'Vectors/FFI_Cadence/{fname}')
		df = None
		if vector_path is not None:
			local = os.path.join(vector_path, fname)
			if os.path.isfile(local):
				try:
					df = pd.read_csv(local, comment='#', index_col=False)
				except Exception:
					df = None
		if df is None:
			try:
				df = pd.read_csv(url, comment='#', index_col=False)
			except Exception:
				df = None

		if df is not None:
			times_btjd = times_mjd - 56999.5
			vec_t = df['MidTime'].values
			orbits = np.zeros(len(times_btjd), dtype=int)
			for segment in np.unique(df['Segment'].values):
				ind = np.where(df['Segment'].values == segment)[0]
				o = (vec_t[ind[0]] - 0.1 <= times_btjd) & (times_btjd <= vec_t[ind[-1]] + 0.1)
				orbits[o] = segment
			return orbits

	# Fallback: split on large time gaps
	diffs = np.diff(times_mjd)
	breaks = np.where(diffs > gap_threshold)[0] + 1
	segments = np.ones(len(times_mjd), dtype=int)
	for k, b in enumerate(breaks, start=2):
		segments[b:] = k
	return segments


def orbit_ref_subtract(flux, times_mjd, sector=None, camera=None,
					   vector_path=None, gap_threshold=0.5):
	"""
	Subtract a per-orbit reference from each orbit's frames.

	For each orbit the reference is the median stack of that orbit's frames,
	subtracted only from frames belonging to that orbit.

	Parameters
	----------
	flux          : ndarray (T, NY, NX)
	times_mjd     : ndarray (T,)
	sector, camera: int — used to fetch TESSVectors orbit assignments
	vector_path   : str or None — local directory for cached TESSVectors files
	gap_threshold : float — fallback gap size in days if TESSVectors unavailable

	Returns
	-------
	result   : ndarray (T, NY, NX)
	segments : ndarray (T,)
	orbit_refs : dict
	"""
	segments = get_orbit_segments(times_mjd, sector=sector, camera=camera,
								  vector_path=vector_path,
								  gap_threshold=gap_threshold)
	orbs = np.unique(segments)

	orbit_refs = {s: np.nanmedian(flux[segments == s], axis=0) for s in orbs}

	result = np.empty_like(flux)
	for s in orbs:
		mask = segments == s
		result[mask] = flux[mask] - orbit_refs[s]

	return result, segments, orbit_refs


def _blend_frame(bkg_new_i, bkg_prev_i, delta_i, resid_prev_i, sigma, sharp_mask_i):
	_, _, scale = sigma_clipped_stats(resid_prev_i)
	w = np.clip(delta_i / (sigma * scale + 1e-10), 0, 1)
	w_zero = (w == 0)
	if w_zero.any():
		w_med = median_filter(w_zero.astype(float), size=7)
		diff_mask = w_zero & ~(w_med > 0.5)
		w[w_med > 0.5] = 0
		if diff_mask.any():
			w[diff_mask] = np.nan
			w = interpolate_replace_nans(w, Gaussian2DKernel(1.0))
	w = np.where(np.isfinite(w), w, 0.0)
	if sharp_mask_i is not None:
		w[sharp_mask_i] = 0.0
	result = (1 - w) * bkg_new_i + w * bkg_prev_i
	result = np.where(np.isfinite(result), result, bkg_prev_i)
	return result

def blend_dynamic_background(bkg_new, bkg_prev, flux, sigma=2.0, sharp_masks=None, n_jobs=1, backend='loky', verbose=0):
	"""Per-pixel blend bkg_new toward bkg_prev based on residual quality.

	For each pixel, compares |flux - bkg_new| vs |flux - bkg_prev|. Where
	bkg_new leaves a larger residual than bkg_prev it is blended back toward
	bkg_prev. The blend is soft: the transition is scaled by the per-frame
	typical residual so noise-level differences do not trigger blending.

	Pixels in sharp_masks always use bkg_new (weight=0) — sharp anomaly
	corrections are trusted unconditionally.

	Parameters
	----------
	bkg_new : np.ndarray (T, NY, NX)
	bkg_prev : np.ndarray (T, NY, NX)
	flux : np.ndarray (T, NY, NX)
	sigma : float
	sharp_masks : np.ndarray (T, NY, NX) bool, optional

	Returns
	-------
	np.ndarray (T, NY, NX)
	"""
	bkg_new = np.array(bkg_new)
	flux = np.array(flux)
	resid_new = np.abs(flux - bkg_new)
	resid_prev = np.abs(flux - bkg_prev)
	delta = resid_new - resid_prev
	T = bkg_new.shape[0]
	sharp_list = [sharp_masks[i] if sharp_masks is not None else None for i in range(T)]

	results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_blend_frame)(bkg_new[i], bkg_prev[i], delta[i], resid_prev[i],
							  sigma, sharp_list[i])
		for i in range(T))
	return np.array(results)

