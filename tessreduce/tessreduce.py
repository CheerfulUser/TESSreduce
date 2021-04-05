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

from sklearn.cluster import OPTICS, cluster_optics_dbscan

from scipy.signal import savgol_filter


from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline

from photutils import centroid_com
from photutils import DAOStarFinder

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip

import multiprocessing
from joblib import Parallel, delayed

from .catalog_tools import *
from .calibration_tools import *
from .rescale_straps import correct_straps

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# set the package directory so we can load in a file later
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'


from astropy.coordinates import SkyCoord
from astropy import units as u


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
		   (thing <= np.percentile(thing[ind],10))) * 1.

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
	if np.nansum(data) > 0:
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
		nearest = griddata((x1, y1), newarr.ravel(),
								  (xx, yy),method='nearest')
		if extrapolate:
			estimate[np.isnan(estimate)] = nearest[np.isnan(estimate)]
		
		estimate = gaussian_filter(estimate,1)
	else:
		estimate = np.zeros_like(data) * np.nan

	return estimate

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
		split = np.where(np.diff(tpf.time.mjd) > 0.5)[0][0] + 1
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

def Lightcurve(flux, aper,zeropoint=20.44, normalise = False):
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
	scale = 'counts'
	for k in range(len(LC)):
		if np.isnan(flux[k]*aper).all():
			LC[k] = np.nan

	if scale.lower() == 'normalise':
		LC = LC / np.nanmedian(LC)
	elif scale.lower() == 'magnitude':
		LC = -2.5*np.log10(LC) + zeropoint
	elif scale.lower() == 'flux':
		LC = -2.5*np.log10(LC) + zeropoint
		#LC = 10**

	return LC


class tessreduce():

	def __init__(self,ra=None,dec=None,name=None,tpf=None,size=90,sector=None,reduce=False,
				 align=True,parallel=True,verbose=1):
		"""
		Class to reduce tess data.
		"""
		self.ra   = ra
		self.dec   = dec 
		self.name   = name
		self.size   = size
		self.align   = align
		self.sector   = sector
		self.verbose   = verbose
		self.parallel   = parallel
		self.calibrate   = False



		# calculated 
		self.mask   = None
		self.shift  = None
		self.bkg	= None
		self.flux   = None
		self.ref	= None
		self.wcs	= None
		self.qe	 = None
		self.lc	 = None
		self.sky	= None
		self.events = None
		self.zp	 = 20.44 # default value 

		if tpf is not None:
			self.flux = strip_units(tpf.flux)
			self.wcs  = tpf.wcs
			self.ra   = tpf.ra
			self.dec  = tpf.dec

		elif self.check_coord():
			self.Get_TESS()

		if reduce:
			self.reduce()


	def check_coord(self):
		if ((self.ra is None) | (self.dec is None)) & (self.name is None):
			return False
		else:
			return True

	def Get_TESS(self,ra=None,dec=None,name=None,Size=None,Sector=None):
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
		if (ra is not None) & (dec is not None):
			c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec) *
						 u.degree, frame='icrs')
		else:
			c = SkyCoord(ra=float(self.ra)*u.degree, dec=float(self.dec) *
						 u.degree, frame='icrs')

		if Sector is None:
			Sector = self.sector
		if self.name is None:
			tess = lk.search_tesscut(c,sector=Sector)
		else:
			tess = lk.search_tesscut(name,sector=Sector)
		if Size is None:
			Size = self.size
		tpf = tess.download(cutout_size=Size)
	
		self.tpf  = tpf
		self.flux = strip_units(tpf.flux)
		self.wcs  = tpf.wcs

	def Make_mask(self,maglim=19,scale=1,strapsize=4):
		data = strip_units(self.flux)

		mask = Cat_mask(self.tpf,maglim,scale,strapsize)
		sources = ((mask & 1)+1 ==1) * 1.
		sources[sources==0] = np.nan
		tmp = np.nansum(data*sources,axis=(1,2))
		tmp[tmp==0] = 1e12 # random big number 
		ref = data[np.argmin(tmp)] * sources
		try:
			qe = correct_straps(ref,mask,parallel=True)
		except:
			qe = correct_straps(ref,mask,parallel=False)
		mm = Source_mask(ref * qe * sources)
		mm[np.isnan(mm)] = 0
		mm = mm.astype(int)
		mm = abs(mm-1)

		fullmask = mask | (mm*1)
		self.mask = fullmask

	def background(self):

		m = (self.mask == 0) * 1.
		m[m==0] = np.nan

		if (self.flux.shape[1] > 30) & (self.flux.shape[2] > 30):
			flux = strip_units(self.flux)

			bkg_smth = np.zeros_like(flux) * np.nan
			if self.parallel:
				num_cores = multiprocessing.cpu_count()
				bkg_smth = Parallel(n_jobs=num_cores)(delayed(Smooth_bkg)(frame) for frame in flux*m)
			else:
				for i in range(flux.shape[0]):
					bkg_smth[i] = Smooth_bkg(flux[i]*m)
		else:
			print('Small tpf, using percentile cut background')
			bkg_smth = self.Small_background()

		mm = (self.mask & ~4) == 0 
		mm[mm==0] = np.nan
		strap = ((self.mask & 4) > 0) * 1.
		strap[strap==0] = np.nan

		data = strip_units(self.flux)
		if type(data) != np.ndarray:
			data = data.value
		qes = np.zeros_like(bkg_smth) * np.nan
		for i in range(data.shape[0]):
			s = (data[i]*strap* mm)/bkg_smth[i]
			q = np.zeros_like(s) * np.nan
			for j in range(s.shape[1]):
				q[:,j] = np.nanmedian(s[:,j])
			q[np.isnan(q)] =1 
			qes[i] = q
		bkg = bkg_smth * qes
		
		self.qe = qes 
		self.bkg = bkg 


	def Small_background(self):
		bkg = np.zeros_like(self.flux)
		flux = strip_units(self.flux)
		lim = np.percentile(flux,10,axis=(1,2))
		ind = flux > lim[:,np.newaxis,np.newaxis]
		flux[ind] = np.nan
		val = np.nanmedian(flux,axis=(1,2))
		bkg[:,:,:] = val[:,np.newaxis,np.newaxis]
		self.bkg = bkg

	def get_ref(self,start = None, stop = None):
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
		data = strip_units(self.flux)
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
		self.ref = reference


	def Centroids_DAO(self):
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
		f = strip_units(self.flux)
		m = self.ref.copy()

		mean, med, std = sigma_clipped_stats(m, sigma=3.0)
		
		daofind = DAOStarFinder(fwhm=3.0, threshold=5.*std)
		s = daofind(m - med)
		mx = s['xcentroid']
		my = s['ycentroid']
		
		if self.parallel:
			
			num_cores = multiprocessing.cpu_count()
			shifts = Parallel(n_jobs=num_cores)(
				delayed(Calculate_shifts)(frame,mx,my,daofind) for frame in f)
			shifts = np.array(shifts)
		else:
			shifts = np.zeros((len(f),2,len(mx))) * np.nan
			for i in range(len(f)):
				shifts[i,:,:] = Calculate_shifts(f[i],mx,my,daofind)


		meds = np.nanmedian(shifts,axis = 2)
		meds[~np.isfinite(meds)] = 0
		
		smooth = Smooth_motion(meds,self.tpf)
		nans = np.nansum(f,axis=(1,2)) ==0
		smooth[nans] = np.nan
		self.shift = smooth





	def Shift_images(self,median=False):
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

		Data = self.flux
		Offset = self.shift

		shifted = Data.copy()
		data = Data.copy()
		data[data<0] = 0
		for i in range(len(data)):
			if np.nansum(data[i]) > 0:
				shifted[i] = shift(data[i],[-Offset[i,1],-Offset[i,0]],mode='nearest',order=3)
		self.flux = shifted




	def bin_data(self,lc=None,time_bin=6/24,frames = None):
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
		if lc is None:
			lc = self.lc
		else:
			if lc.shape[0] > lc.shape[1]:
				lc = lc.T
		flux = lc[1]
		t    = lc[0]
		if time_bin is None:
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
			binlc = np.array([t[x],lc])
		else:
			lc = []
			points = np.arange(t[0]+time_bin*.5,t[-1],time_bin)
			time_inds = abs(points[:,np.newaxis] - t[np.newaxis,:]) <= time_bin/2
			for i in range(len(points)):
				lc += [np.nanmedian(flux[time_inds[i]])]
			lc = np.array(lc)
			binlc = np.array([points,lc])
		return binlc


	def Diff_lc(self,time=None,x=None,y=None,ra=None,dec=None,tar_ap=3,sky_in=5,sky_out=9,plot=False,mask=None):
		data = strip_units(self.flux)
		if ((ra is None) | (dec is None)) & ((x is None) | (y is None)):
			ra = self.ra 
			dec = self.dec

		if tar_ap // 2 == tar_ap / 2:
			print(Warning('tar_ap must be odd, adding 1'))
			tar_ap += 1
		if sky_out // 2 == sky_out / 2:
			print(Warning('sky_out must be odd, adding 1'))
			sky_out += 1
		if sky_in // 2 == sky_in / 2:
			print(Warning('sky_out must be odd, adding 1'))
			sky_in += 1
			
		if (ra is not None) & (dec is not None) & (self.tpf is not None):
			x,y = self.wcs.all_world2pix(ra,dec,0)
			x = int(x + 0.5)
			y = int(y + 0.5)
		elif (x is None) & (y is None):
			x,y = self.wcs.all_world2pix(self.ra,self.dec,0)
			x = int(x + 0.5)
			y = int(y + 0.5)
		ap_tar = np.zeros_like(data[0])
		ap_sky = np.zeros_like(data[0])
		ap_tar[y,x]= 1
		ap_sky[y,x]= 1
		ap_tar = convolve(ap_tar,np.ones((tar_ap,tar_ap)))
		ap_sky = convolve(ap_sky,np.ones((sky_out,sky_out))) - convolve(ap_sky,np.ones((sky_in,sky_in)))
		ap_sky[ap_sky == 0] = np.nan
		m = sigma_clip((self.ref)*ap_sky).mask
		ap_sky[m] = np.nan
		
		temp = np.nansum(data*ap_tar,axis=(1,2))
		ind = temp < np.percentile(temp,40)
		med = np.nanmedian(data[ind],axis=0)
		
		diff = data - self.ref#med
		if mask is not None:
			ap_sky = mask
			ap_sky[ap_sky==0] = np.nan
		sky_med = np.nanmedian(ap_sky*diff,axis=(1,2))
		sky_std = np.nanstd(ap_sky*diff,axis=(1,2))
		
		tar = np.nansum(diff*ap_tar,axis=(1,2))
		tar -= sky_med * tar_ap**2
		tar_err = sky_std * tar_ap**2
		tar[tar_err > 100] = np.nan
		sky_med[tar_err > 100] = np.nan
		if self.tpf is not None:
			time = self.tpf.time.mjd
		lc = np.array([time, tar, tar_err])
		sky = np.array([time, sky_med, sky_std])
		
		if plot:
			self.dif_diag_plot(ap_tar,ap_sky,lc = lc,sky=sky,data=diff)
		
		return lc, sky

	def dif_diag_plot(self,ap_tar,ap_sky,lc=None,sky=None,data=None):
		if lc is None:
			lc = self.lc
		if sky is None:
			sky = self.sky
		if data is None:
			data = self.flux
		plt.figure(figsize=(9,4))
		plt.subplot(121)
		plt.fill_between(lc[0],sky[1]-sky[2],sky[1]+sky[2],alpha=.5,color='C1')
		plt.plot(sky[0],sky[1],'C1.',label='Sky')
		plt.fill_between(lc[0],lc[1]-lc[2],lc[1]+lc[2],alpha=.5,color='C0')
		plt.plot(lc[0],lc[1],'C0.',label='Target')
		binned = self.bin_data(lc=lc)
		plt.plot(binned[0],binned[1],'C2.',label='6hr bin')
		plt.xlabel('MJD')
		plt.ylabel('Counts')
		plt.legend(loc=4)
		plt.subplot(122)
		maxind = np.where((np.nanmax(lc[1]) == lc[1]))[0][0]
		plt.imshow(data[maxind],origin='lower',
				   vmin=np.percentile(data[maxind],16),
				   vmax=np.percentile(data[maxind],99),
				   aspect='auto')
		plt.colorbar()
		ap = ap_tar
		ap[ap==0] = np.nan
		#plt.imshow(ap,origin='lower',alpha = 0.2)
		#plt.imshow(ap_sky,origin='lower',alpha = 0.8,cmap='hot')
		y,x = np.where(ap_sky > 0)
		plt.plot(x,y,'r.',alpha = 0.3)
		
		y,x = np.where(ap > 0)
		plt.plot(x,y,'C1.',alpha = 0.3)

		return

	def plotter(self,lc=None):
		if lc is None:
			lc = self.lc
		plt.figure()
		if lc.shape[0] > lc.shape[1]:
			plt.plot(lc[:,0],lc[:,1])
		else:
			plt.plot(lc[0],lc[1])
		plt.ylabel('Counts')
		plt.xlabel('Time MJD')
		plt.show()
		return

	def reduce(self, aper = None, shift = True, parallel = True, calibrate=False,
						scale = 'counts', bin_size = 0, plot = True, all_output = True,
						mask_scale = 1,diff_lc = True):
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

		scale : str
			options = [counts, magnitude, flux, normalise]
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
		if (self.flux.shape[1] < 30) & (self.flux.shape[2] < 30):
			small = True	
		else:
			small = False

		if small & shift:
			print('Unlikely to get good shifts from a small tpf, so shift has been set to False')
			shift = False

		self.get_ref()
		if self.verbose > 0:
			print('made reference')
		# make source mask
		self.Make_mask(maglim=18,strapsize=4,scale=mask_scale)#Source_mask(ref,grid=0)
		if np.nansum(self.mask) < self.mask.shape[0] * self.mask.shape[1] * .1:
			print('mask is too dense, lowering mask_scale to 0.5')
			self.Make_mask(maglim=18,strapsize=4,scale=0.5)
		if self.verbose > 0:
			print('made source mask')
		# calculate background for each frame
		if self.verbose > 0:
			print('calculating background')
		
		self.background()

		if np.isnan(self.bkg).all():
			# check to see if the background worked
			raise ValueError('bkg all nans')
		
		flux = strip_units(self.flux)

		self.flux = self.flux - self.bkg
		if self.verbose > 0:
			print('background subtracted')
		self.get_ref()
		#return flux, bkg
		if np.isnan(self.flux).all():
			raise ValueError('flux all nans')

		if self.align:
			if self.verbose > 0:
				print('calculating centroids')
			try:
				offset = self.Centroids_DAO()
			except:
				print('Something went wrong, switching to serial')
				self.parallel = False
				self.shift = Centroids_DAO()

			self.Shift_images()
			if self.verbose > 0:
				print('images shifted')
		
		
		

		zp = np.array([20.44,0])
		mask = (self.mask ==0) * 1.
		mask[mask ==0] = np.nan
		err = np.nanmean(mask*self.flux,axis=(1,2))
		if calibrate & (self.dec >= -30):
			zp,err = Calibrate_lc(self.tpf,self.flux)

		elif calibrate & (self.dec < -30):
			print('Target is too far south with Dec = {} for PS1 photometry.'.format(tpf.dec) +
				' Can not calibrate at this time.')

			err = Calculate_err(self.tpf,self.flux)

		if diff_lc:
			self.lc, self.sky = self.Diff_lc(plot=True,sky_in=5,sky_out=9)
		else:
			self.Make_lc(aperture=aper,bin_size=bin_size,
								zeropoint = self.zp,scale=scale)#,normalise=False)
		

	def Make_lc(self,aperture = None,bin_size=0,zeropoint=None,scale='counts',clip = False):
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
		flux = strip_units(self.flux)
		t = self.tpf.time.mjd

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
			 
		lc = Lightcurve(flux,aper)   #,scale = scale)
		if clip:
			mask = ~sigma_mask(lc)
			lc[mask] = np.nan
		if bin_size > 1:
			lc, t = bin_data(t,lc,bin_size)
		lc = np.array([t,lc])
		if (zeropoint is not None) & (scale=='mag'):
			lc[1,:] = -2.5*np.log10(lc[1,:]) + zeropoint
		self.lc = lc

	def lc_events(self,err=None,duration=10,sig=5):
		lc = self.lc
		ind = np.isfinite(lc[1])
		lc = lc[:,ind]
		mask = Cluster_cut(lc,err=err,sig=sig)
		outliers = Identify_masks(mask)
		good = np.nansum(outliers,axis=1) > duration
		outliers = outliers[good]
		print('Found {} events longer than {} frames at {} sigma'.format(outliers.shape[0],duration,sig))
		temp = outliers * lc[1][np.newaxis,:]
		lcs = []
		for event in temp:
			l = (self.lc[:2]).copy()
			l[1,:] = np.nan
			l[1,ind] = event
			lcs += [l]
		lcs = np.array(lcs)
		lcs[lcs == 0] = np.nan
		self.events = lcs

	def event_plotter(self):
		if self.events is None:
			self.lc_events()
		plt.figure()
		plt.plot(self.lc[0],self.lc[1],'k.')
		for i in range(len(self.events)):
			plt.plot(self.events[i,0],self.events[i,1],'*',label='Event {}'.format(i))
		plt.xlabel('MJD')
		plt.ylabel('Counts')


	def detrend_transient(self,lc=None,err=None,Mask=None,variable=False,sig = 5, sig_up = 3, sig_low = 10, tail_length='auto'):
		"""
		Removes all long term stellar variability, while preserving flares. Input a light curve 
		with shape (2,n) and it should work!

		Parameters
		----------
		lc : array
			lightcurve with the shape of (2,n), where the first index is time and the second is 
			flux.
		sig_up : float
			upper sigma clip value 
		sig_low : float
			lower sigma clip value
		tail_length : str OR int
			option for setting the buffer zone of points after the peak. If it is 'auto' it 
			will be determined through functions, but if its an int then it will take the given 
			value as the buffer tail length for fine tuning.

		Outputs
		-------
		trends : array
			the stellar trends, subtract this from your input lc
		"""
		# Make a smoothing value with a significant portion of the total 
		
		if lc is None:
			lc = self.lc[:2]
		nonan = np.isfinite(lc[1])
		lc = lc[:,nonan]

		if (err is None) & (self.lc.shape[0] > 2):
			err = self.lc[2]
			err = err[nonan]

		trends = np.zeros(lc.shape[1])
		break_inds = Multiple_day_breaks(lc)
		#lc[Mask] = np.nan

		if variable:
			size = int(lc.shape[1] * 0.1)
			if size % 2 == 0: size += 1

			finite = np.isfinite(lc[1])
			smooth = savgol_filter(lc[1,finite],size,1)		
			# interpolate the smoothed data over the missing time values
			f1 = interp1d(lc[0,finite], smooth, kind='linear',fill_value='extrapolate')
			smooth = f1(lc[0])
			lc2 = lc.copy()
			lc2[1] = lc2[1] - smooth
			try:

				mask = Cluster_cut(lc2,err=err,sig=sig)
			except:
				print('could not cluster')
				mask = sig_err(lc2[1],err,sig=sig)
			#sigma_clip(lc[1]-smooth,sigma=sig,sigma_upper=sig_up,
			#					sigma_lower=sig_low,masked=True).mask
		else:
			try:
				mask = Cluster_cut(lc,err=err,sig=sig)
			except:
				print('could not cluster')
				mask = sig_err(lc[1],err,sig=sig)

		ind = np.where(mask)[0]
		masked = lc.copy()
		# Mask out all peaks, with a lead in of 5 frames and tail of 100 to account for decay
		# todo: use findpeaks to get height estimates and change the buffers accordingly
		if type(tail_length) == str:
			if tail_length == 'auto':
				#m = auto_tail(lc,mask,err)
				masked[:,mask] = np.nan


			else:
				if lc.shape[1] > 4000:
					tail_length = 100
					start_length = 1
				else:
					tail_length = 10
				for i in ind:
					masked[:,i-5:i+tail_length] = np.nan
		else:
			tail_length = int(tail_length)
			if type(tail_length) != int:
				raise ValueError("tail_length must be either 'auto' or an integer")
			for i in ind:
				masked[:,i-5:i+tail_length] = np.nan


		## Hack solution doesnt need to worry about interpolation. Assumes that stellar variability 
		## is largely continuous over the missing data regions.
		#f1 = interp1d(lc[0,finite], lc[1,finite], kind='linear',fill_value='extrapolate')
		#interp = f1(lc[0,:])

		# Smooth the remaining data, assuming its effectively a continuous data set (no gaps)
		size = int(lc.shape[1] * 0.005)
		if size % 2 == 0: 
			size += 1
		for i in range(len(break_inds)-1):
			section = lc[:,break_inds[i]:break_inds[i+1]]

			mask_section = masked[:,break_inds[i]:break_inds[i+1]]
			if np.nansum(mask_section) < 10:
				mask_section[1,:] = np.nanmedian(masked[1,:])
				if np.nansum(mask_section) < 10:
					mask_section[1,:] = np.nanmedian(section)
			
			if np.isnan(mask_section[1,0]):
				mask_section[1,0] = np.nanmedian(mask_section[1])
			if np.isnan(mask_section[1,-1]):
				mask_section[1,-1] = np.nanmedian(mask_section[1])
			finite = np.isfinite(mask_section[1])
			smooth = savgol_filter(mask_section[1,finite],size,1)

			# interpolate the smoothed data over the missing time values
			f1 = interp1d(section[0,finite], smooth, kind='linear',fill_value='extrapolate')
			trends[break_inds[i]:break_inds[i+1]] = f1(section[0])
			
		# huzzah, we now have a trend that should remove stellar variability, excluding flares.
		detrend = deepcopy(self.lc)
		detrend[1,nonan] -= trends
		return detrend

	def detrend_stellar_var(self,lc=None,err=None,Mask=None,variable=False,sig = None, sig_up = 5, sig_low = 10, tail_length=''):
		"""
		Removes all long term stellar variability, while preserving flares. Input a light curve 
		with shape (2,n) and it should work!
		Parameters
		----------
		lc : array
			lightcurve with the shape of (2,n), where the first index is time and the second is 
			flux.
		sig_up : float
			upper sigma clip value 
		sig_low : float
			lower sigma clip value
		tail_length : str OR int
			option for setting the buffer zone of points after the peak. If it is 'auto' it 
			will be determined through functions, but if its an int then it will take the given 
			value as the buffer tail length for fine tuning.
		Outputs
		-------
		trends : array
			the stellar trends, subtract this from your input lc
		"""
		# Make a smoothing value with a significant portion of the total 
		if lc is None:
			lc = self.lc[:2]
		nonan = np.isfinite(lc[1])
		lc = lc[:,nonan]

		if (err is None) & (self.lc.shape[0] > 2):
			err = self.lc[2]
			err = err[nonan]

		trends = np.zeros(lc.shape[1])
		break_inds = Multiple_day_breaks(lc)
		#lc[Mask] = np.nan
		
		if variable:
			size = int(lc.shape[1] * 0.04)
			if size % 2 == 0: size += 1

			finite = np.isfinite(lc[1])
			smooth = savgol_filter(lc[1,finite],size,1)		
			# interpolate the smoothed data over the missing time values
			f1 = interp1d(lc[0,finite], smooth, kind='linear',fill_value='extrapolate')
			smooth = f1(lc[0])
			mask = sig_err(lc[1]-smooth,err,sig=sig)
			#sigma_clip(lc[1]-smooth,sigma=sig,sigma_upper=sig_up,
			#					sigma_lower=sig_low,masked=True).mask
		else:
			mask = sig_err(lc[1],err,sig=sig)
			
		ind = np.where(mask)[0]
		masked = lc.copy()
		# Mask out all peaks, with a lead in of 5 frames and tail of 100 to account for decay
		# todo: use findpeaks to get height estimates and change the buffers accordingly
		if type(tail_length) == str:
			if tail_length == 'auto':
				
				m = auto_tail(lc,mask,err)
				masked[:,~m] = np.nan
				
				
			else:
				if lc.shape[1] > 4000:
					tail_length = 100
					start_length = 1
				else:
					tail_length = 10
				for i in ind:
					masked[:,i-5:i+tail_length] = np.nan
		else:
			tail_length = int(tail_length)
			if type(tail_length) != int:
				raise ValueError("tail_length must be either 'auto' or an integer")
			for i in ind:
				masked[:,i-5:i+tail_length] = np.nan
		
		
		## Hack solution doesnt need to worry about interpolation. Assumes that stellar variability 
		## is largely continuous over the missing data regions.
		#f1 = interp1d(lc[0,finite], lc[1,finite], kind='linear',fill_value='extrapolate')
		#interp = f1(lc[0,:])

		# Smooth the remaining data, assuming its effectively a continuous data set (no gaps)
		size = int(lc.shape[1] * 0.005)
		if size % 2 == 0: 
			size += 1
		for i in range(len(break_inds)-1):
			section = lc[:,break_inds[i]:break_inds[i+1]]
			finite = np.isfinite(masked[1,break_inds[i]:break_inds[i+1]])
			smooth = savgol_filter(section[1,finite],size,1)
			
			# interpolate the smoothed data over the missing time values
			f1 = interp1d(section[0,finite], smooth, kind='linear',fill_value='extrapolate')
			trends[break_inds[i]:break_inds[i+1]] = f1(section[0])
		# huzzah, we now have a trend that should remove stellar variability, excluding flares.
		detrend = deepcopy(lc)
		detrend[1,:] = lc[1,:] - trends
		return detrend


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






def Calculate_err(tpf,flux):

	tab = Unified_catalog(tpf,magnitude_limit=18)
	if len(tab)> 10:
		col = tab.col.values + .5
		row = tab.row.values + .5
		pos = np.array([col,row]).T

		median = np.nanmedian(flux,axis=0)

		index, med_cut, stamps = Isolated_stars(pos,tab['tmag'].values,flux,median,Distance=3)

		isolated = tab.iloc[index]
		ps1ind = np.isfinite(isolated['imag'].values)

		isolated = isolated.iloc[ps1ind]
		med_cut = med_cut[ps1ind]
		stamps = stamps[ps1ind]
		isolc = np.nansum(stamps,axis=(2,3))
		ind = ((np.nanmedian(isolc,axis=1) > 100) & (np.nanmedian(isolc,axis=1)*.1 >= np.nanstd(isolc,axis=1)) 
				& (np.nanmedian(isolc,axis=1) < 1000))
		isolc = isolc[ind]
		isolated = isolated[ind]
		if len(isolated) < 10:
			warnings.warn('Only {} sources used for zerpoint calculation. Errors may be larger than reported'.format(len(isolated)))
		err = np.nanstd(isolc-np.nanmedian(isolc,axis=1)[:,np.newaxis],axis=0)
		return err
	else:
		warnings.warn('No reference cataloge sources to isolate stars. Can not calculate error with this method')


def Calibrate_lc(tpf,flux,ID=None,diagnostic=False,ref='z',fit='tess'):
	"""

	"""
	if ID is None:
		ID = tpf.targetid
	tab = Unified_catalog(tpf,magnitude_limit=18)
	col = tab.col.values + .5
	row = tab.row.values + .5
	pos = np.array([col,row]).T

	median = np.nanmedian(flux,axis=0)

	index, med_cut, stamps = Isolated_stars(pos,tab['tmag'].values,flux,median,Distance=3)

	isolated = tab.iloc[index]
	ps1ind = np.isfinite(isolated['imag'].values)

	isolated = isolated.iloc[ps1ind]
	med_cut = med_cut[ps1ind]
	stamps = stamps[ps1ind]
	isolc = np.nansum(stamps,axis=(2,3))
	ind = (np.nanmedian(isolc,axis=1) > 100) & (np.nanmedian(isolc,axis=1)*.1 >= np.nanstd(isolc,axis=1)) & (np.nanmedian(isolc,axis=1) < 1000)
	isolc = isolc[ind]
	isolated = isolated[ind]
	if len(isolated) < 10:
		warnings.warn('Only {} sources used for zerpoint calculation. Errors may be larger than reported'.format(len(isolated)))
	err = np.nanstd(isolc-np.nanmedian(isolc,axis=1)[:,np.newaxis],axis=0)
	higherr = sigma_clip(err,sigma=2).mask

	if diagnostic:
		plt.figure()
		plt.title('Isolated reference stars')
		for i in range(len(isolc)):
			plt.plot(-2.5*np.log10(isolc[i]))
			plt.ylabel('System magnitude')
			plt.xlabel('Frame number')
			plt.minorticks_on()

	isolated = Reformat_df(isolated)
	# column names here are just to conform with the calibration code 
	isolated['tessMeanPSFMag'] = -2.5*np.log10(np.nanmedian(isolc[:,~higherr],axis=1))
	# need to do a proper accounting of errors.
	isolated['tessMeanPSFMagErr'] = .1
	try:
		#return(isolated)
		if diagnostic:
			extinction, good_sources = Tonry_reduce(isolated,plot=True)
		else: 
			extinction, good_sources = Tonry_reduce(isolated,plot=False)

		model = np.load(package_directory+'calspec_mags.npy',allow_pickle=True).item()

		compare_ref = np.array([['g-r','r-'+ref],['g-r','i-'+ref],['g-r','y-'+ref],['g-r','g-i']])
		compare_fit = np.array([['g-r','r-'+fit],['g-r',fit+'-y'],['g-r',fit+'-i'],['g-r',fit+'-z']])

		zp_ref, d_ref = Fit_zeropoint(good_sources,model,compare_ref,extinction,ref)
		zp_fit, d_fit = Fit_zeropoint(good_sources,model,compare_fit,extinction,fit)

		if diagnostic:
			c_fit = Make_colours(d_fit,model,compare_fit,Extinction = extinction)
			zeropointPlotter(zp_fit,zp_ref,c_fit,compare_fit,ID,fit,'figs/'+ID,Close=False)
			zeropointPlotter(zp_fit,zp_ref,c_fit,compare_fit,ID,fit,'figs/'+ID,Residuals=True,Close=False)

		zero_point = zp_fit
		zero_point_err = zp_ref
		zp = np.array([zero_point, zero_point_err])
	except:
		zp = np.array([20.44, 0])
	return zp, err

### Serious source mask

def Cat_mask(tpf,maglim=19,scale=1,strapsize=3,badpix=None):
	from .cat_mask import Big_sat, gaia_auto_mask, ps1_auto_mask, Strap_mask
	wcs = tpf.wcs
	image = tpf.flux[100]
	image = strip_units(image)

	gp,gm = Get_Gaia(tpf,magnitude_limit=maglim)
	gaia  = pd.DataFrame(np.array([gp[:,0],gp[:,1],gm]).T,columns=['x','y','mag'])
	if tpf.dec > -30:
		pp,pm = Get_PS1(tpf,magnitude_limit=maglim)
		ps1   = pd.DataFrame(np.array([pp[:,0],pp[:,1],pm]).T,columns=['x','y','mag'])
		mp  = ps1_auto_mask(ps1,image,scale)
	else:
		mp = {}
		mp['all'] = np.zeros_like(image)
	
	sat = Big_sat(gaia,image,scale)
	mg  = gaia_auto_mask(gaia,image,scale)
	

	sat = (np.nansum(sat,axis=0) > 0).astype(int) * 2 # assign 2 bit 
	mask = ((mg['all']+mp['all']) > 0).astype(int) * 1 # assign 1 bit
	if strapsize > 0: 
		strap = Strap_mask(image,tpf.column,strapsize).astype(int) * 4 # assign 4 bit 
	else:
		strap = np.zeros_like(image,dtype=int)
	if badpix is not None:
		bp = cat_mask.Make_bad_pixel_mask(badpix, file)
		totalmask = mask | sat | strap | bp
	else:
		totalmask = mask | sat | strap
	
	return totalmask

	


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

def Cluster_cut(lc,err=None,sig=3,smoothing=True,buffer=24):
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




### Difference imaging

	
	
