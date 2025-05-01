import os
import requests
import json
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from io import BytesIO

from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from scipy.ndimage import label
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from sklearn.cluster import OPTICS
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
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
from tess_stars2px import tess_stars2px_function_entry as focal_plane
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
	data = deepcopy(data)
	data[mask] = np.nan
	estimate = inpaint.inpaint_biharmonic(data,mask)
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
				if np.nanmedian(estimate) < 20:
					gauss_smooth = gauss_smooth * 3
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
			shifts[0,:] = np.nan
			shifts[1,:] = np.nan
	return shifts

def image_sub(theta, image, ref):
	dx, dy = theta
	s = shift(image,([dx,dy]),order=5)
	#translation = np.float64([[1,0,dx],[0,1, dy]])
	#s = cv2.warpAffine(image, translation, image.shape[::-1], flags=cv2.INTER_CUBIC,borderValue=0)
	diff = (ref-s)**2
	if image.shape[0] > 50:
		return np.nansum(diff[10:-11,10:-11])
	else:
		return np.nansum(diff[5:-6,5:-6])

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
		bds = [(-1,1),(-1,1)]
		res = minimize(image_sub,x0,args=(image,ref),method = 'Powell',bounds= bds)
		s = res.x
	else:
		s = np.zeros((2)) * np.nan
	if (s == np.ones((2))).any():
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
	skernel = int(len(tpf.flux) * 0.2) #simple way of making the smoothing window 10% of the duration
	skernel = skernel // 2 +1
	print('!!! skernel '+ str(skernel))
	#skernel = 25
	if skernel < 25:
		skernel = 25
	try:
		try:
			split = np.where(np.diff(tpf.time.mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
			ind2 = np.where(ind2 != 0)[0] + split
			smoothed[ind1,0] = savgol_filter(Centroids[ind1,0],skernel,3)
			smoothed[ind2,0] = savgol_filter(Centroids[ind2,0],skernel,3)

			smoothed[ind1,1] = savgol_filter(Centroids[ind1,1],skernel,3)
			smoothed[ind2,1] = savgol_filter(Centroids[ind2,1],skernel,3)
		except:
			split = np.where(np.diff(tpf.time.mjd) > 0.5)[0][0] + 1
			# ugly, but who cares
			ind1 = np.nansum(tpf.flux[:split],axis=(1,2))
			ind1 = np.where(ind1 != 0)[0]
			ind2 = np.nansum(tpf.flux[split:],axis=(1,2))
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

		new_ind = [i for i in ind if i < len(sec_times)]

		secs = sec_times.iloc[new_ind]
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

	outID, outEclipLong, outEclipLat, outSecs, outCam, outCcd, outColPix, \
	outRowPix, scinfo = focal_plane(0, ra, dec)
	
	sec_times = pd.read_csv(package_directory + 'sector_mjd.csv')
	if len(outSecs) > 0:
		ind = outSecs - 1 

		new_ind = [i for i in ind if i < len(sec_times)]

		secs = sec_times.iloc[new_ind]
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

def par_psf_flux(image,prf,shift=[0,0],bkg_poly_order=3,kernel=None):
	if np.isnan(shift)[0]:
		shift = np.array([0,0])
	prf.psf_flux(image,ext_shift=shift,poly_order=bkg_poly_order,kernel=kernel)
	return prf.flux, prf.eflux

def par_psf_full(cutout,prf,shift=[0,0],xlim=0.5,ylim=0.5):
	if np.isnan(shift)[0]:
		shift = np.array([0,0])
	prf.psf_position(cutout,ext_shift=shift,limx=xlim,limy=ylim)
	prf.psf_flux(cutout)
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

def regional_stats_mask(image,size=90,sigma=3,iters=10):
	if size < 30:
		print('!!! Region size is small !!!')
	sx, sy = image.shape
	X, Y = np.ogrid[0:sx, 0:sy]
	regions = sy//size * (X//size) + Y//size
	max_reg = np.max(regions)

	clip = np.zeros_like(image)
	for i in range(max_reg+1):
		rx,ry = np.where(regions == i)
		m,me, s = sigma_clipped_stats(image[ry,rx],maxiters=iters)
		cut_ind = np.where((image[rx,ry] >= me+sigma*s) | (image[rx,ry] <= me-sigma*s))
		clip[rx[cut_ind],ry[cut_ind]] = 1
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

	b_obj_size = []
	for i in range(b_objects):
		b_obj_size += [np.sum(b_labeled==i)]
	b_obj_size = np.array(b_obj_size)

	a_obj_size = []
	for i in range(a_objects):
		a_obj_size += [np.sum(a_labeled==i)]
	a_obj_size = np.array(a_obj_size)

	for i in range(a_objects):
		if (a_obj_size[i] >= max_size) | (a_obj_size[i] <= 9):
			a_labeled[a_labeled==i] = 0

	for i in range(b_objects):
		if (b_obj_size[i] >= max_size) | (b_obj_size[i] <= 9):
			b_labeled[b_labeled==i] = 0
			
			
	overlap = (a_labeled>0) & (b_labeled>0)
	y,x = np.where(overlap)


	good_a = np.unique(a_labeled[y,x])
	good_b = np.unique(b_labeled[y,x])

	a_ratio = []
	for ind in good_a:
		eh = a_labeled == ind
		eh2 = eh * overlap
		ratio = np.sum(eh2) / np.sum(eh)
		a_ratio += [ratio]
	a_ratio = np.array(a_ratio)
		
		
	b_ratio = []
	for ind in good_b:
		eh = b_labeled == ind 
		eh2 = eh * overlap
		ratio = np.sum(eh2) / np.sum(eh)
		b_ratio += [ratio]
	b_ratio = np.array(b_ratio)


	for i in good_a[a_ratio<0.2]: 
		a_labeled[a_labeled==i] = 0
	for i in good_b[b_ratio<0.2]: 
		b_labeled[b_labeled==i] = 0
	c = (a_labeled + b_labeled) > 0

	c_labeled, c_objects = label(c==0) 
	for i in range(c_objects):
		if np.sum(c_labeled==i) < 10:
			c[c_labeled==i] = 1
	
	#points = fftconvolve(c,np.ones((5,5)),mode='same')
	points = c>0#oints > 0.8
	data = deepcopy(bkg)
	data[points] = np.nan
	estimate = inpaint.inpaint_biharmonic(data,points)
	return estimate

