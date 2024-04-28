import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from photutils.detection import StarFinder

from copy import deepcopy
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from skimage.restoration import inpaint

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.optimize import minimize

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip

from tess_stars2px import tess_stars2px_function_entry as focal_plane
from tabulate import tabulate

package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time


import requests
import json

def strip_units(data):
	if type(data) != np.ndarray:
		data = data.value
	return data

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


def parallel_bkg3(data,mask):
	new_data = deepcopy(data)
	new_data[mask] = np.nan
	estimate = inpaint.inpaint_biharmonic(new_data,mask)
	return estimate

def Smooth_bkg(data, gauss_smooth=2, interpolate=False, extrapolate=True):
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
				
				estimate = gaussian_filter(estimate,gauss_smooth)
			
			#estimate = median_filter(estimate,5)
			else:
				# try inpaint stuff 
				mask = deepcopy(arr.mask)
				mask = mask.astype(bool)
				# end inpaint
				estimate = inpaint.inpaint_biharmonic(data,mask)
				#estimate = signal.fftconvolve(estimate,self.prf,mode='same')
				estimate = gaussian_filter(estimate,gauss_smooth)
		else:
			estimate = np.zeros_like(data) * np.nan	
	else:
		estimate = np.zeros_like(data) #* np.nan	

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
			shifts[0,indo] = np.nan
			shifts[1,indo] = np.nan
	return shifts

def image_sub(theta, image, ref):
	dx, dy = theta
	s = shift(image,([dx,dy]),order=5)
	#translation = np.float64([[1,0,dx],[0,1, dy]])
	#s = cv2.warpAffine(image, translation, image.shape[::-1], flags=cv2.INTER_CUBIC,borderValue=0)
	diff = (ref-s)**2
	return np.nansum(diff[20:-20,20:-20])

def difference_shifts(image,ref):
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
		bds = [(-2,2),(-2,2)]
		res = minimize(image_sub,x0,args=(image,ref),method = 'Powell')#,bounds= bds)
		s = res.x
	else:
		s = np.zeros((2)) * np.nan
	return s

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
		try:
			split = np.where(np.diff(tpf.time.mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
			ind2 = np.where(ind2 != 0)[0] + split
			smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],25,3)
			smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],25,3)

			smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],25,3)
			smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],25,3)
		except:
			split = np.where(np.diff(tpf.time.mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
			ind2 = np.where(ind2 != 0)[0] + split
			smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],11,3)
			smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],11,3)

			smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],11,3)
			smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],11,3)

	except IndexError:
		smoothed[:,0] = savgol_filter(Centroids[:,0],25,3)		
		smoothed[:,1] = savgol_filter(Centroids[:,1],25,3)
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
	rad = np.sqrt(np.gradient(flux)**2+np.gradient(np.gradient(flux))**2)
	return rad

def grad_flux_rad(flux):
	rad = np.sqrt(flux**2+np.gradient(flux)**2)
	return rad


def sn_lookup(name,time='disc',buffer=0,print_table=True):
	"""
	Check for overlapping TESS ovservations for a transient. Uses the Open SNe Catalog for 
	discovery/max times and coordinates.

	------
	Inputs
	------
	name : str
		catalog name
	time : str
		reference time to use, can be either disc, or max
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
	try:
		url = 'https://api.astrocats.space/{}'.format(name)
		response = requests.get(url)
		json_acceptable_string = response.content.decode("utf-8").replace("'", "").split('\n')[0]
		d = json.loads(json_acceptable_string)
		if list(d.keys())[0] == 'message':
			#print(d['message'])

			#return None
			tns = True
		else:
			disc_t = d[name]['discoverdate'][0]['value']
			disc_t = Time(disc_t.replace('/','-'))
			max_t = d[name]['maxdate'][0]['value']
			max_t = Time(max_t.replace('/','-'))
			ra = d[name]['ra'][-1]['value']
			dec = d[name]['dec'][-1]['value']
			tns = False
	except:
		tns = True
	if tns:
		#print('!! Open SNe Catalog down, using TNS !!')
		name = name[name.index('2'):]
		url = f'https://www.wis-tns.org/object/{name}' # hard coding in that the event is in the 2000s
		headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
		result = requests.get(url, headers=headers)
		if result.ok:
			ra, dec = result.text.split('<div class="alter-value">')[-1].split('<')[0].split(' ')
			ra = float(ra); dec = float(dec)
			disc_t = Time(result.text.split('<span class="name">Discovery Date</span><div class="value"><b>')[-1].split('<')[0])
			max_t = deepcopy(disc_t)

			c = SkyCoord(ra,dec, unit=(u.hourangle, u.deg))
			ra = c.ra.deg
			dec = c.dec.deg

	c = SkyCoord(ra,dec, unit=(u.hourangle, u.deg))
	ra = c.ra.deg
	dec = c.dec.deg
	
	outID, outEclipLong, outEclipLat, outSecs, outCam, outCcd, outColPix, \
	outRowPix, scinfo = focal_plane(0, ra, dec)
	
	sec_times = pd.read_csv(package_directory + 'sector_mjd.csv')
	if len(outSecs) > 0:
		ind = outSecs - 1 
		secs = sec_times.iloc[ind]
		if (time.lower() == 'disc') | (time.lower() == 'discovery'):
			disc_start = secs['mjd_start'].values - disc_t.mjd
			disc_end = secs['mjd_end'].values - disc_t.mjd
		elif (time.lower() == 'max') | (time.lower() == 'peak'):
			disc_start = secs['mjd_start'].values - max_t.mjd
			disc_end = secs['mjd_end'].values - max_t.mjd

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
		return tr_list
	else:
		print('No TESS coverage')
		return None

def spacetime_lookup(ra,dec,time=None,buffer=0,print_table=True):
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

	outID, outEclipLong, outEclipLat, outSecs, outCam, outCcd, outColPix, \
	outRowPix, scinfo = focal_plane(0, ra, dec)
	
	sec_times = pd.read_csv(package_directory + 'sector_mjd.csv')
	if len(outSecs) > 0:
		ind = outSecs - 1 
		secs = sec_times.iloc[ind]
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

def par_psf_flux(image,prf,shift=[0,0]):
	if np.isnan(shift)[0]:
		shift = np.array([0,0])
	prf.psf_flux(image,ext_shift=shift)
	return prf.flux