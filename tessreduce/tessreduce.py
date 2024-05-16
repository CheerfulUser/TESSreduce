"""
Import packages!
"""
import traceback
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import lightkurve as lk
from photutils.detection import StarFinder
from PRF import TESS_PRF

from copy import deepcopy

from scipy.ndimage.filters import convolve
from scipy.ndimage import shift

from sklearn.cluster import OPTICS

from scipy.signal import savgol_filter


from scipy.interpolate import interp1d

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy import wcs


import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

from .catalog_tools import *
from .calibration_tools import *
from .ground_tools import ground
from .rescale_straps import correct_straps
from .lastpercent import *
from .psf_photom import create_psf
from .helpers import *
from .cat_mask import Big_sat, gaia_auto_mask, ps1_auto_mask, Strap_mask
#from .syndiff import PS1_scene

# turn off runtime warnings (lots from logic on nans)
import warnings
# nuke warnings because sigma clip is extremely annoying 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
pd.options.mode.chained_assignment = None
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	sigma_clip
	sigma_clipped_stats

# set the package directory so we can load in a file later
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'


from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.time import Time


fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27			   # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches


class tessreduce():

	def __init__(self,ra=None,dec=None,name=None,obs_list=None,tpf=None,size=90,sector=None,reduce=True,
				 align=True,parallel=True,diff=True,plot=False,corr_correction=True,phot_method='aperture',savename=None,
				 quality_bitmask='default',verbose=1,cache_dir=None,calibrate=True,harshmask_counts=None,
				 sourcehunt=True,num_cores='max',catalogue_path=None):
		"""
		Class to reduce tess data.
		"""
		self.ra = ra
		self.dec = dec 
		self.name = name
		self.size = size
		self.align = align
		self.sector = sector
		self.verbose = verbose
		self.parallel = parallel
		self.calibrate = calibrate
		self.corr_correction = corr_correction
		self.diff = diff
		self.tpf = tpf
		self._assign_phot_method(phot_method)
		self._harshmask_counts = harshmask_counts
		self._sourcehunt = sourcehunt
		if catalogue_path is None:
			catalogue_path = os.getcwd()
		self._catalogue_path = catalogue_path
		if type(num_cores) == str:
			self.num_cores = multiprocessing.cpu_count()
		else:
			self.num_cores = num_cores


		# Plotting
		self.plot = plot
		self.savename = savename

		# calculated 
		self.mask = None
		self.shift = None
		self.bkg = None
		self.flux = None
		self.ref = None
		self.ref_ind = None
		self.wcs = None
		self.qe = None
		self.lc = None
		self.sky = None
		self.events = None
		self.zp = None
		self.zp_e = None
		self.sn_name = None
		self.ebv = 0
		# repeat for backup
		self.tzp = None
		self.tzp_e = None
		
		# light curve units 
		self.lc_units = 'Counts'


		if obs_list is not None:
			obs_list = np.array(obs_list,dtype=object)
			if len(obs_list.shape) > 1:
				obs_list = obs_list[obs_list[:,3].astype('bool')][0]
			self.ra = obs_list[0]
			self.dec = obs_list[1]
			self.sector = obs_list[2]

		if tpf is not None:
			if type(tpf) == str:
				self.tpf = lk.TessTargetPixelFile(tpf)
			self.flux = strip_units(self.tpf.flux)
			self.wcs = self.tpf.wcs
			self.ra = self.tpf.ra
			self.dec = self.tpf.dec
			self.size = self.tpf.flux.shape[1]

		elif self.check_coord():
			if self.verbose>0:
				print('getting TPF from TESScut')
			self.get_TESS(quality_bitmask=quality_bitmask,cache_dir=cache_dir)
			#self.tpf = external_get_TESS()
			self.flux = strip_units(self.tpf.flux)
			self.wcs  = self.tpf.wcs

		self.ground = ground(ra = self.ra, dec = self.dec)
		self._get_gaia()

		if reduce:
			self.reduce()


	def check_coord(self):
		if ((self.ra is None) | (self.dec is None)) & (self.name is None):
			return False
		else:
			return True

	def _get_gaia(self,maglim=21):
		result = Get_Catalogue(self.tpf, Catalog = 'gaia')
		result = result[result.Gmag < maglim]
		result = result.rename(columns={'RA_ICRS': 'ra',
                               'DE_ICRS': 'dec',
                               'e_RA_ICRS': 'e_ra',
                               'e_DE_ICRS': 'e_dec',})
		x,y = self.wcs.all_world2pix(result['ra'].values,result['dec'].values,0)
		result['x'] = x; result['y'] = y
		ind = (((x > 0) & (y > 0)) & 
		 	  ((x < (self.flux.shape[2])) & (y < (self.flux.shape[1]))))
		result = result.iloc[ind]

		self.gaia = result
		

	def _assign_phot_method(self,phot_method):
		if type(phot_method) == str:
			method = phot_method.lower()
			if (method == 'psf') | (method == 'aperture'):
				self.phot_method = method
			else:
				m = f'The input method "{method}" is not supported, please select either "psf", or "aperture".'
				raise ValueError(m)
		else:
			m = 'phot_mehtod must be a string equal to either "psf", or "aperture".'
			raise ValueError(m)

	def get_TESS(self,ra=None,dec=None,name=None,Size=None,Sector=None,quality_bitmask='default',cache_dir=None):
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
		if Sector is None:
			Sector = self.sector

		if (name is None) & (self.name is None):
			if (ra is not None) & (dec is not None):
				c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec) *
							 u.degree, frame='icrs')
			else:
				c = SkyCoord(ra=float(self.ra)*u.degree, dec=float(self.dec) *
							 u.degree, frame='icrs')
			tess = lk.search_tesscut(c,sector=Sector)
		else:
			tess = lk.search_tesscut(name,sector=Sector)
		if Size is None:
			Size = self.size
		
		tpf = tess.download(quality_bitmask=quality_bitmask,cutout_size=Size,download_dir=cache_dir)
		if tpf is None:
			m = 'Failure in TESScut api, not sure why.'
			raise ValueError(m)
		self.tpf  = tpf
		self.flux = strip_units(tpf.flux)
		self.wcs  = tpf.wcs

	def harsh_mask(self):
		if self._harshmask_counts is not None:
			ind = self.ref > self._harshmask_counts
			self.ref[ind]

	def make_mask(self,catalogue_path=None,maglim=19,scale=1,strapsize=6,useref=False):
		# make a diagnostic plot for mask
		data = strip_units(self.flux)
		if useref:
			mask, cat = Cat_mask(self.tpf,catalogue_path,maglim,scale,strapsize,ref=self.ref)
		else:
			mask, cat = Cat_mask(self.tpf,catalogue_path,maglim,scale,strapsize)
		sky = ((mask & 1)+1 ==1) * 1.
		sky[sky==0] = np.nan
		tmp = np.nansum(data*sky,axis=(1,2))
		tmp[tmp==0] = 1e12 # random big number 
		ref = data[np.argmin(tmp)] * sky
		try:
			qe = correct_straps(ref,mask,parallel=True)
		except:
			qe = correct_straps(ref,mask,parallel=False)
		#mm = Source_mask(ref * qe * sky)
		#mm[np.isnan(mm)] = 0
		#mm = mm.astype(int)
		#mm = abs(mm-1)
		#block out center 
		c1 = data.shape[1] // 2
		c2 = data.shape[2] // 2
		cmask = np.zeros_like(data[0],dtype=int)
		cmask[c1,c2] = 1
		kern = np.ones((5,5))
		cmask = convolve(cmask,kern)
		#mm = (mm*1) | cmask

		fullmask = mask | cmask
		sky = ((fullmask & 1)+1 ==1) * 1.
		sky[sky==0] = np.nan
		masked = ref*sky
		mean = np.nanmean(masked) # assume sources weight the mean above the bkg
		if useref is False:
			m_second = (masked > mean).astype(int)
			self.mask = fullmask | m_second
		else:
			self.mask = fullmask
		self._mask_cat = cat

	def psf_source_mask(self,mask,sigma=5):
		
		
		prf = TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,
				   	   self.tpf.column+self.flux.shape[2]/2,self.tpf.row+self.flux.shape[1]/2)
		#prf_directory = '/fred/oz100/_local_TESS_PRFs'
		#if self.sector < 4:
		#	prf = TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,
		#					self.tpf.column+self.flux.shape[2]/2,self.tpf.row+self.flux.shape[1]/2,
		#					localdatadir=f'{prf_directory}/Sectors1_2_3')
		#else:
		#	prf = TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,
		#					self.tpf.column+self.flux.shape[2]/2,self.tpf.row+self.flux.shape[1]/2,
		#					localdatadir=f'{prf_directory}/Sectors4+')
		self.prf =  prf.locate(5,5,(11,11))

		
		data = (self._flux_aligned - self.ref) #* mask
		if self.parallel:
			try:
				m = Parallel(n_jobs=self.num_cores)(delayed(par_psf_source_mask)(frame,self.prf,sigma) for frame in data)
				m = np.array(m)
			except:
				m = np.ones_like(data)
				for i in range(data.shape[0]):
					#m[i] = _par_psf_source_mask(data[i],self.prf,sigma)
					eh = par_psf_source_mask(data[i],self.prf,sigma)
					m[i] = eh
		else:
			m = np.ones_like(data)
			for i in range(data.shape[0]):
				m[i] = par_psf_source_mask(data[i],self.prf,sigma)
		return m * 1.0

	def background(self,calc_qe=True, strap_iso=True,source_hunt=False,gauss_smooth=2,interpolate=True):
		"""
		Calculate the background for all frames in the TPF.
		"""
		if strap_iso:
			m = (self.mask == 0) * 1.
		else:
			m = ((self.mask & 1 == 0) & (self.mask & 2 == 0) ) * 1.
		m[m==0] = np.nan

		if source_hunt:
			sm = self.psf_source_mask(m)
			sm[sm==0] = np.nan
			m = sm * m
		self._bkgmask = m

		if (self.flux.shape[1] > 30) & (self.flux.shape[2] > 30):
			flux = strip_units(self.flux)

			bkg_smth = np.zeros_like(flux) * np.nan
			if self.parallel:
				bkg_smth = Parallel(n_jobs=self.num_cores)(delayed(Smooth_bkg)(frame,gauss_smooth,interpolate) for frame in flux*m)
			else:
				for i in range(flux.shape[0]):
					bkg_smth[i] = Smooth_bkg((flux*m)[i],gauss_smooth,interpolate)
		else:
			print('Small tpf, using percentile cut background')
			self.small_background()
			bkg_smth = self.bkg
		
		if calc_qe:
			strap = (self.mask == 4) * 1.0
			strap[strap==0] = np.nan
			# check if its a time varying mask
			if len(strap.shape) == 3: 
				strap = strap[self.ref_ind]
			mask = ((self.mask & 1) == 0) * 1.0
			mask[mask==0] = np.nan
		
			data = strip_units(self.flux) * mask
			norm = self.flux / bkg_smth
			straps = norm * ((self.mask & 4)>0)
			limit = np.nanpercentile(straps,60,axis=1)
			straps[limit[:,np.newaxis,:] < straps] = np.nan
			straps[straps==0] = 1

			value = np.nanmedian(straps,axis=1)

			qe = np.ones_like(bkg_smth) * value[:,np.newaxis,:]
			bkg = bkg_smth * qe
			self.qe = qe
		else:
			bkg = np.array(bkg_smth)
		self.bkg = bkg 


	def small_background(self):
		bkg = np.zeros_like(self.flux)
		flux = strip_units(self.flux)
		lim = 2*np.nanmin(flux,axis=(1,2))#np.nanpercentile(flux,1,axis=(1,2))
		ind = flux > lim[:,np.newaxis,np.newaxis]
		flux[ind] = np.nan
		val = np.nanmedian(flux,axis=(1,2))
		bkg[:,:,:] = val[:,np.newaxis,np.newaxis]
		self.bkg = bkg

	def _bkg_round_3(self,iters=5):
		for i in range(iters):
			tb = self.bkg * self._bkgmask
			m = np.nanmedian(tb,axis=(1,2))
			std = np.nanstd(tb,axis=(1,2))
			sbkg = np.nansum(self.bkg,axis=(1,2))
			ind = sbkg > np.nanpercentile(sbkg,95)

			frame,y,x = np.where((self.bkg>(2*std+m)[:,np.newaxis,np.newaxis]) | (self.bkg<(m - 2*std)[:,np.newaxis,np.newaxis]))

			dist_mask = np.zeros_like(self.flux)

			dist_mask[frame,y,x] = 1
			dist_mask[ind] = 0 # reset the bright frames since they are unreliable 
			common = np.sum(dist_mask,axis=0) > len(dist_mask) * 0.3
			dist_mask[:,common] = 1
			kern = np.ones((1,3,3))
			dist_mask = convolve(dist_mask,kern) > 0
			if self.parallel:
				bkg_3 = Parallel(n_jobs=self.num_cores)(delayed(parallel_bkg3)(self.bkg[i],dist_mask[i]) 
														   for i in np.arange(len(dist_mask)))
			else:
				bkg_3 = []
				bkg_smth = np.zeros_like(dist_mask)
				for i in range(len(dist_mask)):
					bkg_3[i] = parallel_bkg3(self.bkg[i],dist_mask[i])
			self.bkg = np.array(bkg_3)

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
		data = strip_units(self.flux)
		if (start is None) & (stop is None):
			start = 0
			stop = len(self.flux)
		elif (start is not None) & (stop is None):
			stop = len(self.flux)

		elif (start is None) & (stop is not None):
			start = 0

		start = int(start)
		stop = int(stop)

		ind = self.tpf.quality[start:stop] == 0
		d = deepcopy(data[start:stop])[ind]
		summed = np.nansum(d,axis=(1,2))
		lim = np.percentile(summed[np.isfinite(summed)],5)
		summed[summed>lim] = 0
		inds = np.where(ind)[0]
		ref_ind = start + inds[np.argmax(summed)]
		reference = data[ref_ind]
		if len(reference.shape) > 2:
			reference = reference[0]
			ref_ind = ref_ind[0]
		
		self.ref = reference
		self.ref_ind = ref_ind


	def centroids_DAO(self,plot=None,savename=None):
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
		if plot is None:
			plot = self.plot
		if savename is None:
			savename = self.savename
		# hack solution for new lightkurve
		f = strip_units(self.flux)
		m = self.ref.copy()

		mean, med, std = sigma_clipped_stats(m, sigma=3.0)

		prf = TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,
				   	   self.tpf.column+self.flux.shape[2]/2,self.tpf.row+self.flux.shape[1]/2)
		self.prf =  prf.locate(5,5,(11,11))
		
		finder = StarFinder(2*std,kernel=self.prf,exclude_border=True)
		s = finder.find_stars(m-med)
		#daofind = DAOStarFinder(fwhm=2.0, threshold=10.*std,exclude_border=True)
		#s = daofind(m - med)
		mx = s['xcentroid']
		my = s['ycentroid']
		x_mid = self.flux.shape[2] / 2
		y_mid = self.flux.shape[1] / 2
		#ind = #((abs(mx - x_mid) <= 30) & (abs(my - y_mid) <= 30) & 
		#ind = (abs(mx - x_mid) >= 5) & (abs(my - y_mid) >= 5)
		#self._dat_sources = s[ind].to_pandas()
		self._dat_sources = s.to_pandas()
		#mx = mx[ind]
		#my = my[ind]
		if self.parallel:
			
			
			shifts = Parallel(n_jobs=self.num_cores)(
				delayed(Calculate_shifts)(frame,mx,my,finder) for frame in f)
			shifts = np.array(shifts)
		else:
			shifts = np.zeros((len(f),2,len(mx))) * np.nan
			for i in range(len(f)):
				shifts[i,:,:] = Calculate_shifts(f[i],mx,my,finder)

		self.raw_shifts = shifts
		meds = np.nanmedian(shifts,axis = 2)
		meds[~np.isfinite(meds)] = 0

		smooth = Smooth_motion(meds,self.tpf)
		nans = np.nansum(f,axis=(1,2)) ==0
		smooth[nans] = np.nan
		self.shift = meds#smooth
		if plot:
			#meds[meds==0] = np.nan
			t = self.tpf.time.mjd
			ind = np.where(np.diff(t) > .5)[0]
			smooth[ind,:] = np.nan
			plt.figure(figsize=(1.5*fig_width,1*fig_width))
			plt.plot(t,meds[:,1],'.',label='Row shift',alpha =0.5)
			plt.plot(t,smooth[:,1],'-',label='Smoothed row shift')
			plt.plot(t,meds[:,0],'.',label='Col shift',alpha =0.5)
			plt.plot(t,smooth[:,0],'-',label='Smoothed col shift')
			#plt.plot(thing,'+')
			plt.ylabel('Shift (pixels)',fontsize=15)
			plt.xlabel('Time (MJD)',fontsize=15)
			plt.legend()
			#plt.tight_layout()
			plt.show()
			if savename is not None:
				plt.savefig(savename+'_disp.pdf', bbox_inches = "tight")
		
	def fit_shift(self,plot=None,savename=None):
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
		if plot is None:
			plot = self.plot
		if savename is None:
			savename = self.savename
		
		sources = ((self.mask & 1) ==1) * 1.0 - (self.mask & 2)
		sources[sources<=0] = 0

		f = self.flux #* sources[np.newaxis,:,:]
		m = self.ref.copy() * sources
		m[m==0] = np.nan
		if self.parallel:

			shifts = Parallel(n_jobs=self.num_cores)(
				delayed(difference_shifts)(frame,m) for frame in f)
			shifts = np.array(shifts)
		else:
			shifts = np.zeros((len(f),2)) * np.nan
			for i in range(len(f)):
				shifts[i,:] = difference_shifts(f[i],m)


		#smooth = Smooth_motion(meds,self.tpf)
		#nans = np.nansum(f,axis=(1,2)) ==0
		#smooth[nans] = np.nan
		if self.shift is not None:
			self.shift += shifts
		else:
			self.shift = shifts
		if plot:
			#meds[meds==0] = np.nan
			t = self.tpf.time.mjd
			ind = np.where(np.diff(t) > .5)[0]
			shifts[ind,:] = np.nan
			plt.figure(figsize=(1.5*fig_width,1*fig_width))
			plt.plot(t,shifts[:,0],'.',label='Row shift',alpha =0.5)
			plt.plot(t,shifts[:,1],'.',label='Col shift',alpha =0.5)

			plt.ylabel('Shift (pixels)',fontsize=15)
			plt.xlabel('Time (MJD)',fontsize=15)
			plt.legend()
			#plt.tight_layout()
			plt.show()
			if savename is not None:
				plt.savefig(savename+'_disp_corr.pdf', bbox_inches = "tight")



	def shift_images(self,median=False):
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
		shifted = self.flux.copy()
		nans = ~np.isfinite(shifted)
		shifted[nans] = 0.
		if median:
			for i in range(len(shifted)):
				if np.nansum(abs(shifted[i])) > 0:
					shifted[i] = shift(self.ref,[-self.shift[i,1],-self.shift[i,0]])
			self.flux -= shifted
		else:
			for i in range(len(shifted)):
				if np.nansum(abs(shifted[i])) > 0:
					#translation = np.float64([[1,0,self.shift[i,0]],[0,1, self.shift[i,1]]])
					#shifted[i] = cv2.warpAffine(shifted[i], translation, shifted[i].shape[::-1], flags=cv2.INTER_CUBIC,borderValue=0)
					shifted[i] = shift(shifted[i],[self.shift[i,0],self.shift[i,1]],mode='nearest',order=5)#mode='constant',cval=np.nan)
			#shifted[0,:] = np.nan
			#shifted[-1,:] = np.nan
			#shifted[:,0] = np.nan
			#shifted[:,-1] = np.nan
			self.flux = shifted
			

				#print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
		#shifted[nans] = np.nan



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
		try:
			err = lc[2]
		except:
			err = deepcopy(lc[1]) * np.nan
		t	= lc[0]
		if time_bin is None:
			bin_size = int(frames)
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
			
			points = np.arange(t[0]+time_bin*.5,t[-1],time_bin)
			time_inds = abs(points[:,np.newaxis] - t[np.newaxis,:]) <= time_bin/2
			l = []
			e = []
			for i in range(len(points)):
				l += [np.nanmedian(flux[time_inds[i]])]
				e += [np.nanmedian(err[time_inds[i]])]
			l = np.array(l)
			e = np.array(e)
			binlc = np.array([points,l,e])
		return binlc


	def bin_flux(self,flux=None,time_bin=6/24,frames = None):
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
		if flux is None:
			flux = self.flux

		t = self.tpf.time.mjd

		if time_bin is None:
			bin_size = int(frames)
			f = []
			x = []
			for i in range(int(len(flux)/bin_size)):
				if np.isnan(flux[i*bin_size:(i*bin_size)+bin_size]).all():
					f.append(np.nan)
					x.append(int(i*bin_size+(bin_size/2)))
				else:
					f.append(np.nanmedian(flux[i*bin_size:(i*bin_size)+bin_size],axis=0))
					x.append(int(i*bin_size+(bin_size/2)))
			binf = np.array(f)
			bint = np.array(x)
		else:
			
			points = np.arange(t[0]+time_bin*.5,t[-1],time_bin)
			time_inds = abs(points[:,np.newaxis] - t[np.newaxis,:]) <= time_bin/2
			f = []
			for i in range(len(points)):
				f += [np.nanmedian(flux[time_inds[i]],axis=0)]
			binf = np.array(f)
			bint = np.array(points)
		return binf, bint

	def diff_lc(self,time=None,x=None,y=None,ra=None,dec=None,tar_ap=3,
				sky_in=5,sky_out=9,phot_method=None,plot=None,savename=None,mask=None,diff = True):
		"""
		Calculate the difference imaged light curve. if no position is given (x,y or ra,dec)
		then it degaults to the centre. Sky flux is calculated with an annulus aperture surrounding 
		the target aperture and subtracted from the source. The sky aperture undergoes sigma clipping
		to remove pixels that are poorly subtracted and contain other sources.

		------
		Inputs
		------
			time : array
				1d array of times 
			x : int 
				centre of target aperture in x dim 
			y : int 
				centre of target aperture in y dim
			ra : float
				centre of target aperture in ra
			dec : float
				centre of target aperture in dec
			tar_ap : int (odd)
				width of the aperture
			sky_in : int (odd)
				inner edge of the sky aperture 
			sky_out : int (odd, larger than sky_in)
				outter edge of the sky aperture 
			plot : bool
				option for plotting diagnostic plot
			mask : array
				optional sky mask 

		------
		Output
		------
			lc : array (3xn)
				difference imaged light curve of target. 
				lc[0] = time, lc[1] = flux, lc[2] = flux error

			sky : array (3xn)
				difference imaged light curve of sky. 
				sky[0] = time, sky[1] = flux, sky[2] = flux error
				
		"""
		if plot is None:
			plot = self.plot
		if savename is None:
			savename = self.savename
		if phot_method is None:
			phot_method = self.phot_method

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
		m = sigma_clip((self.ref)*ap_sky,sigma=1).mask
		ap_sky[m] = np.nan
		
		temp = np.nansum(data*ap_tar,axis=(1,2))
		ind = temp < np.percentile(temp,40)
		med = np.nanmedian(data[ind],axis=0)
		med = np.nanmedian(data,axis=0)
		if not diff:
			data = data + self.ref
		if mask is not None:
			ap_sky = mask
			ap_sky[ap_sky==0] = np.nan
		mean_sky, sky_med, sky_std = sigma_clipped_stats(ap_sky*data,axis=(1,2))
		#sky_med = np.nanmedian(ap_sky*data,axis=(1,2))
		#sky_std = np.nanstd(ap_sky*data,axis=(1,2))
		if phot_method == 'aperture':
			if self.diff:
				tar = np.nansum(data*ap_tar,axis=(1,2))
			else:
				tar = np.nansum((data+self.ref)*ap_tar,axis=(1,2))
			tar -= sky_med * tar_ap**2
			tar_err = sky_std * tar_ap**2
		if phot_method == 'psf':
			tar = self.psf_photometry(x,y,diff=diff)
			tar_err = sky_std # still need to work this out
		#tar[tar_err > 100] = np.nan
		#sky_med[tar_err > 100] = np.nan
		if self.tpf is not None:
			time = self.tpf.time.mjd
		lc = np.array([time, tar, tar_err])
		sky = np.array([time, sky_med, sky_std])
		
		if plot:
			self.dif_diag_plot(ap_tar,ap_sky,lc = lc,sky=sky,data=data)
			plt.show()
			if savename is not None:
				plt.savefig(savename + '_diff_diag.pdf', bbox_inches = "tight")
		return lc, sky

	def dif_diag_plot(self,ap_tar,ap_sky,lc=None,sky=None,data=None):
		"""
		Makes a plot showing the target light curve, sky, and difference image at the brightest point
		in the target lc.

		------
		Inputs
		------
			ap_tar : array
				aperture mask
			ap_sky : array
				sky mask
			data : array (shape = 3)
				sequence of images

		------
		Output
		------
			Figure
		"""
		if lc is None:
			lc = self.lc
		if sky is None:
			sky = self.sky
		if data is None:
			data = self.flux
		plt.figure(figsize=(3*fig_width,1*fig_width))
		plt.subplot(121)
		plt.fill_between(lc[0],sky[1]-sky[2],sky[1]+sky[2],alpha=.5,color='C1')
		plt.plot(sky[0],sky[1],'C1.',label='Sky')
		plt.fill_between(lc[0],lc[1]-lc[2],lc[1]+lc[2],alpha=.5,color='C0')
		plt.plot(lc[0],lc[1],'C0.',label='Target')
		binned = self.bin_data(lc=lc)
		plt.plot(binned[0],binned[1],'C2.',label='6hr bin')
		plt.xlabel('Time (MJD)',fontsize=15)
		plt.ylabel('Flux ($e^-/s$)',fontsize=15)
		plt.legend(loc=4)

		plt.subplot(122)
		ap = ap_tar
		ap[ap==0] = np.nan
		maxind = np.where((np.nanmax(lc[1]) == lc[1]))[0]
		try:
			maxind = maxind[0]
		except:
			pass
		d = data[maxind]
		nonan1 = np.isfinite(d)
		nonan2 = np.isfinite(d*ap)
		plt.imshow(data[maxind],origin='lower',
				   vmin=np.nanpercentile(d,16),
				   vmax=np.nanpercentile(d[nonan2],80),
				   aspect='auto')
		cbar = plt.colorbar()
		cbar.set_label('$e^-/s$',fontsize=15)
		plt.xlabel('Column',fontsize=15)
		plt.ylabel('Row',fontsize=15)
		
		#plt.imshow(ap,origin='lower',alpha = 0.2)
		#plt.imshow(ap_sky,origin='lower',alpha = 0.8,cmap='hot')
		y,x = np.where(ap_sky > 0)
		plt.plot(x,y,'r.',alpha = 0.3)
		
		y,x = np.where(ap > 0)
		plt.plot(x,y,'C1.',alpha = 0.3)

		return

	def plotter(self,lc=None,ax = None,ground=False,time_bin=6/24,xlims=None):
		"""
		Simple plotter for light curves. 

		------
		Inputs (Optional)
		------
		lc : np.array
			light curve with dimensions of at least [2,n]
		ax : matplotlib axes
			existing figure axes to add data to 
		time_bin : float
			time range to bin data to in days. ie 1 = 24 hours.
		-------
		Options
		-------
			ground : bool
				if True then ground based data is plotted alongside TESS
		"""
		if ground:
			if self.ground.ztf is None:
				self.ground.get_ztf_data()
			if self.lc_units.lower() == 'counts':
				self.to_flux()

		if lc is None:
			lc = self.lc
		av = self.bin_data(lc=lc,time_bin=time_bin)
		if time_bin * 24 == int(time_bin * 24):
			lab = int(time_bin * 24) 
			
		else:
			lab = time_bin *24

		if ax is None:
			plt.figure(figsize=(1.5*fig_width,1*fig_width))
			ax = plt.gca()
		if lc.shape[0] > lc.shape[1]:
			ax.plot(lc[:,0],lc[:,1],'k.',alpha = 0.4,ms=1,label='$TESS$')
			
			ax.plot(av[:,0],av[:,1],'k.',label='$TESS$ {}hr'.format(lab))
		else:
			ax.plot(lc[0],lc[1],'.k',alpha = 0.4,ms=1,label='$TESS$')
			ax.plot(av[0],av[1],'.k',label='$TESS$ {}hr'.format(lab))
		
		if self.lc_units == 'AB mag':
			ax.invert_yaxis()
			if ground & (self.ground.ztf is not None):
				gind = self.ground.ztf.fid.values == 'g'
				rind = self.ground.ztf.fid.values == 'r'
				ztfg = self.ground.ztf.iloc[gind]
				ztfr = self.ground.ztf.iloc[rind]
				ax.scatter(ztfg.mjd,ztfg.maglim,c='C2',s=.5,alpha = 0.6,marker='v',label='ZTF g non-detec')
				ax.scatter(ztfr.mjd,ztfr.maglim,c='r',s=.5,alpha = 0.6,marker='v',label='ZTF r non-detec')

				ax.errorbar(ztfg.mjd, ztfg.mag,yerr = ztfg.mag_e, c='C2', fmt='o', ms= 5, label='ZTF g')
				ax.errorbar(ztfr.mjd, ztfr.mag,yerr = ztfr.mag_e, c='r', fmt='o', ms=5, label='ZTF r')
				ax.set_ylabel('Apparent magnitude',fontsize=15)
		else:
			ax.set_ylabel('Flux (' + self.lc_units + ')',fontsize=15)
			if ground & (self.ground.ztf is not None):
				self.ground.to_flux(flux_type=self.lc_units)
				gind = self.ground.ztf.fid.values == 'g'
				rind = self.ground.ztf.fid.values == 'r'
				ztfg = self.ground.ztf.iloc[gind]
				ztfr = self.ground.ztf.iloc[rind]
				ax.scatter(ztfg.mjd,ztfg.fluxlim,c='C2',alpha = 0.6,s=20,marker='v',label='ZTF g non-detec')
				ax.scatter(ztfr.mjd,ztfr.fluxlim,c='r',alpha = 0.6,s=20,marker='v',label='ZTF r non-detec')

				ax.errorbar(ztfg.mjd, ztfg.flux,yerr = ztfg.flux_e,ms=4, c='C2', fmt='o', label='ZTF g')
				ax.errorbar(ztfr.mjd, ztfr.flux,yerr = ztfr.flux_e, ms=4, c='r', fmt='o', label='ZTF r')

		if xlims is not None:
			try:
				xmin, xmax = xlims
			except:
				m = 'xlim must have have shape 2 which are MJD times'
				raise ValueError(m)
			plt.xlim(xmin,xmax)

			ind = (lc[0] < xmax) & (lc[0] > xmin)

			ymin = np.nanmin(lc[1,ind])
			ymax = np.nanmax(lc[1,ind])

			plt.ylim(1.2*ymin,1.2*ymax)


		ax.set_xlabel('Time (MJD)',fontsize=15 )
		ax.legend()
		return

	def save_lc(self,filename,time_bin=None):
		"""
		Saves the current lightcurve out to csv format, doesn't include flux units.
		
		------
		Inputs
		------
		filename : str
			output name of the file
		time_bin : float
			Duration in days for binning the lightcurve. If none, no binning is done.



		
		"""

		if time_bin is not None:
			l = self.bin_data(time_bin=time_bin)
		else:
			l = self.lc

		lc = self.to_lightkurve(lc = l)
		format='csv'
		filename = filename.split('.csv')[0]
		if format == 'csv':
			lc.to_csv(filename)
		


	def to_lightkurve(self,lc=None,flux_unit=None):
		"""
		Convert TESSreduce light curve into lighkurve.lightcurve object. Flux units are recorded
		
		-----------------
		Inputs (optional)
		-----------------
		lc : array
			light curve with 2xn or 3xn shape
		flux_unit : str
			units of the light curve flux 
			Valid options:
				counts
				mjy
				cgs
		-------
		Returns
		-------
		light : lightcurve
			lightkurve lightcurve object. All lk function will work on this!
		"""
		if lc is None:
			lc = self.lc
		if flux_unit is None:
			flux_unit = self.lc_units
		if flux_unit.lower() == 'counts':
			unit = u.electron/ u.s
		elif flux_unit.lower() == 'mjy':
			unit = 1e-3 * u.Jy
		elif flux_unit.lower() == 'jy':
			unit = u.Jy
		elif flux_unit.lower() == 'cgs':
			unit = u.erg/u.s/u.cm**2/u.Hz
		else:
			unit = 1
		if lc.shape[0] == 3:
			light = lk.LightCurve(time=Time(lc[0], format='mjd'),flux=lc[1] * unit,flux_err=lc[2] * unit)
		else:
			light = lk.LightCurve(time=Time(lc[0], format='mjd'),flux=lc[1] * unit)
		return light

	def _update_reduction_params(self,align,parallel,calibrate,plot,diff_lc,diff,verbose,
								 corr_correction):
		"""
		Updates relevant parameters for if reduction functions are called out of order.
		"""
		if align is not None:
			self.align = align
		if parallel is not None:
			self.parallel = parallel
		if verbose is not None:
			self.verbose = verbose
		if calibrate is not None:
			self.calibrate = calibrate
		if diff is not None:
			self.diff = diff
		if corr_correction is not None:
			self.corr_correction = corr_correction


	def correlation_corrector(self,limit=0.8):
		"""
		A final corrector that removes the final ~0.5% of the background from pixels that have been 
		interpolated over. Assuning the previously calculated background is a reasonable estimate 
		of what the background is like this function finds the coefficient that when multiplied to the 
		background and subtracted from the flux minimises the correlation between the background and 
		the pixel light curve. This function saves the correlation correction as corr_coeff, and 
		applies the correction to the flux, for all pixels that aren't included as sky pixels. 
			This process seems to do a good job at removing some of the residual background structure 
		that is present in some pixels. 

		"""
		flux, bkg = multi_correlation_cor(self,limit=limit,cores=self.num_cores)
		self.flux = flux 
		self.bkg = bkg


	def _psf_initialise(self,cutoutSize,loc,time_ind=None,ref=False):
		"""
		For gathering the cutouts and PRF base.
		"""
		if time_ind is None:
			time_ind = np.arange(0,len(self.flux))

		if (type(loc[0]) == float) | (type(loc[0]) == np.float64) |  (type(loc[0]) == np.float32):
			loc[0] = int(loc[0]+0.5)
		if (type(loc[1]) == float) | (type(loc[1]) == np.float64) |  (type(loc[1]) == np.float32):
			loc[1] = int(loc[1]+0.5)
		col = self.tpf.column - int(self.size/2-1) + loc[0] # find column and row, when specifying location on a *say* 90x90 px cutout
		row = self.tpf.row - int(self.size/2-1) + loc[1] 
			
		prf = TESS_PRF(self.tpf.camera,self.tpf.ccd,self.tpf.sector,col,row) # initialise psf kernel
		if ref:
			cutout = (self.flux+self.ref)[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
		else:
			cutout = self.flux[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
		return prf, cutout

	def moving_psf_photometry(self,xpos,ypos,size=5,time_ind=None):
		if time_ind is None:
			if len(xpos) != len(self.flux):
				m = 'If "times" is not specified then xpos must have the same length as flux.'
				raise ValueError(m)
			else:
				time_ind = np.arange(0,len(flux))
			if (len(xpos) != len(time_ind)) | (len(ypos) != len(time_ind)):
				m = 'xpos/ypos and time_ind must be the same length'
				raise ValueError(m)
		inds = np.arange(0,len(xpos))
		if self.parallel:
			prfs, cutouts = zip(*Parallel(n_jobs=self.num_cores)(delayed(par_psf_initialise)(self.flux,self.tpf.camera,self.tpf.ccd,
																   						     self.tpf.sector,self.tpf.column,self.tpf.row,
																						     cutoutSize,[xpos[i],ypos[i]],time_ind) for i in inds))
		else:
			prfs = []
			cutouts = []
			for i in range(len(time_ind)):
				prf, cutout = self._psf_initialise(size,[xpos[i],ypos[i]],time_ind=time_ind[i])
				prfs += [prf]
				cutouts += [cutout]
		cutouts = np.array(cutouts)
		print('made cutouts')
		if self.parallel:
			flux, pos = zip(*Parallel(n_jobs=self.num_cores)(delayed(par_psf_full)(cutouts[i],prfs[i],self.shift[i]) for i in inds))
		else:
			flux = []
			pos = []
			for i in range(len(xpos)):
				f, p = par_psf_full(cutouts[i],prfs[i],self.shift[i])
				flux += [f]
				pos += [p]
		flux = np.array(flux)
		pos = np.array(pos)
		pos[0,:] += xpos; pos[1,:] += ypos
		return flux, pos


	def psf_photometry(self,xPix,yPix,size=5,snap='brightest',ext_shift=True,plot=False,diff=None):
		"""
		Main Function! Just switch self to self inside tessreduce and all should follow.

		--------
		Inputs:
		
		self : tessreduce object
		xPix : x pixel location of target region
		yPix : y pixel location of target region
		size : size of cutout to use (should be odd)
		repFact : super sampling factor for modelling
		
		--------
		Options:

		snap : Determines how psf position is fit.
			- None = each frame's position will be fit and used when fitting for flux
			- 'brightest' = the position of the brightest cutout frame will be applied to all subsequent frames
			- int = providing an integer allows for explicit choice of which frame to use as position reference
			- 'ref' = use the reference as the position fit point

		--------
		Returns:

		flux : flux light curve across entire sector.

		"""
		if diff is None:
			diff = self.diff
		flux = []

		if isinstance(xPix,(list,np.ndarray)):
			self.moving_psf_phot()

		else:
			if snap == None:  # if no snap, each cutout has their position fitted and considered during flux fitting
				prf, cutouts = self._psf_initialise(size,(xPix,yPix))   # gather base PRF and the array of cutouts data
				xShifts = []
				yShifts = []
				for cutout in tqdm(cutouts):
					PSF = create_psf(prf,size)
					PSF.psf_position(cutout)
					PSF.psf_flux(cutout)
					flux.append(PSF.flux)
					yShifts.append(PSF.source_y)
					xShifts.append(PSF.source_x)
				if plot:
					fig,ax = plt.subplots(ncols=3,figsize=(12,4))
					ax[0].plot(flux)
					ax[0].set_ylabel('Flux')
					ax[1].plot(xShifts,marker='.',linestyle=' ')
					ax[1].set_ylabel('xShift')
					ax[2].plot(yShifts,marker='.',linestyle=' ')
					ax[2].set_ylabel('yShift')

			elif type(snap) == str:
				if snap == 'brightest': # each cutout has position snapped to brightest frame fit position
					prf, cutouts = self._psf_initialise(size,(xPix,yPix),ref=(not diff))   # gather base PRF and the array of cutouts data
					ind = np.where(cutouts==np.nanmax(cutouts))[0][0]
					ref = cutouts[ind]
					base = create_psf(prf,size)
					base.psf_position(ref,ext_shift=self.shift[ind])
				elif snap == 'ref':
					prf, cutouts = self._psf_initialise(size,(xPix,yPix),ref=True)   # gather base PRF and the array of cutouts data
					ref = cutouts[self.ref_ind]
					base = create_psf(prf,size)
					base.psf_position(ref)
					if diff:
						_, cutouts = self._psf_initialise(size,(xPix,yPix),ref=False)
				if self.parallel:
					inds = np.arange(len(cutouts))
					flux = Parallel(n_jobs=self.num_cores)(delayed(par_psf_flux)(cutouts[i],base,self.shift[i]) for i in inds)
				else:
					for i in range(len(cutouts)):
						flux += [par_psf_flux(cutouts[i],base,self.shift[i])]
				if plot:
					plt.figure()
					plt.plot(flux)
					plt.ylabel('Flux')

			
			elif type(snap) == int:	   # each cutout has position snapped to 'snap' frame fit position (snap is integer)
				base = create_psf(prf,size)
				base.psf_position(cutouts[snap])
				for cutout in cutouts:
					PSF = create_psf(prf,size)
					PSF.source_x = base.source_x
					PSF.source_y = base.source_y
					PSF.psf_flux(cutout)
					flux.append(PSF.flux)
				if plot:
					fig,ax = plt.subplots(ncols=1,figsize=(12,4))
					ax.plot(flux)
					ax.set_ylabel('Flux')
			flux = np.array(flux)
		return flux




	def reduce(self, aper = None, align = None, parallel = None, calibrate=None,
				bin_size = 0, plot = None, mask_scale = 1, ref_start=None, ref_stop=None,
				diff_lc = None,diff=None,verbose=None, tar_ap=3,sky_in=7,sky_out=11,
				moving_mask=None,mask=None,double_shift=False,corr_correction=None,test_seed=None):
		"""
		Reduce the images from the target pixel file and make a light curve with aperture photometry.
		This background subtraction method works well on tpfs > 50x50 pixels.
		
		----------
		Parameters 
		----------
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
		
		-------
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
		try:
			self._update_reduction_params(align, parallel, calibrate, plot, diff_lc, diff, verbose,corr_correction)

			if (self.flux.shape[1] < 30) & (self.flux.shape[2] < 30):
				small = True	
			else:
				small = False

			if small & self.align:
				print('Unlikely to get good shifts from a small tpf, so shift has been set to False')
				self.align = False

			self.get_ref(ref_start,ref_stop)
			if self.verbose > 0:
				print('made reference')
			# make source mask
			if mask is None:
				self.make_mask(catalogue_path=self._catalogue_path,maglim=18,strapsize=7,scale=mask_scale)#Source_mask(ref,grid=0)
				frac = np.nansum((self.mask == 0) * 1.) / (self.mask.shape[0] * self.mask.shape[1])
				#print('mask frac ',frac)
				if frac < 0.05:
					print('!!!WARNING!!! mask is too dense, lowering mask_scale to 0.5, and raising maglim to 15. Background quality will be reduced.')
					self.make_mask(catalogue_path=self._catalogue_path,maglim=15,strapsize=7,scale=0.5)
				if self.verbose > 0:
					print('made source mask')
			else:
				self.mask = mask
				if self.verbose > 0:
					print('assigned source mask')
			# calculate background for each frame
			if self.verbose > 0:
				print('calculating background')
			# calculate the background
			self.background()

			if np.isnan(self.bkg).all():
				# check to see if the background worked
				raise ValueError('bkg all nans')
			
			flux = strip_units(self.flux)
			# subtract background from unitless flux
			self.flux = flux - self.bkg
			# get a ref with low background
			self.ref = deepcopy(self.flux[self.ref_ind])
			if self.verbose > 0:
				print('background subtracted')
			
			
			if np.isnan(self.flux).all():
				raise ValueError('flux all nans')

			if self.align:
				if self.verbose > 0:
					print('Aligning images')
				
				try:
					#self.centroids_DAO()
					#if double_shift:
					#self.shift_images()
					#self.ref = deepcopy(self.flux[self.ref_ind])
					self.fit_shift()
					#self.shift_images()
					
				except:
					print('Something went wrong, switching to serial')
					self.parallel = False
					#self.centroids_DAO()
					self.fit_shift()
					#self.fit_shift()
				#self.fit_shift()
			else:
				self.shift = np.zeros((len(self.flux),2))
			
			if not self.diff:
				if self.align:
					self.shift_images()
					self.flux[np.nansum(self.tpf.flux.value,axis=(1,2))==0] = np.nan
					if self.verbose > 0:
						print('images shifted')

			if self.diff:
				if self.verbose > 0:
					print('!!Re-running for difference image!!')
				# reseting to do diffim 
				self._flux_aligned = deepcopy(self.flux)
				self.flux = strip_units(self.tpf.flux)
				self.flux = self.flux / self.qe

				if self.align:
					self.shift_images()

					if self.verbose > 0:
						print('shifting images')

				if test_seed is not None:
					self.flux += test_seed
				self.flux[np.nansum(self.tpf.flux.value,axis=(1,2))==0] = np.nan
				# subtract reference
				self.ref = deepcopy(self.flux[self.ref_ind])
				self.flux -= self.ref

				self.ref -= self.bkg[self.ref_ind]
				# remake mask
				self.make_mask(catalogue_path=self._catalogue_path,maglim=18,strapsize=7,scale=mask_scale*.8,useref=True)#Source_mask(ref,grid=0)
				frac = np.nansum((self.mask== 0) * 1.) / (self.mask.shape[0] * self.mask.shape[1])
				#print('mask frac ',frac)
				if frac < 0.05:
					print('!!!WARNING!!! mask is too dense, lowering mask_scale to 0.5, and raising maglim to 15. Background quality will be reduced.')
					self.make_mask(catalogue_path=self._catalogue_path,maglim=15,strapsize=7,scale=0.5)
				# assuming that the target is in the centre, so masking it out 
				#m_tar = np.zeros_like(self.mask,dtype=int)
				#m_tar[self.ref.shape[0]//2,self.ref.shape[1]//2]= 1
				#m_tar = convolve(m_tar,np.ones((5,5)))
				#self.mask = self.mask | m_tar
				if moving_mask is not None:
					moving_mask = moving_mask > 0
					temp = np.zeros_like(self.flux,dtype=int)
					temp[:,:,:] = self.mask
					self.mask = temp | moving_mask

				if self.verbose > 0:
					print('remade mask')
				# background
				if self.verbose > 0:
					print('background')
				self.bkg_orig = deepcopy(self.bkg)
				self.background(calc_qe = False,strap_iso = False,source_hunt=self._sourcehunt,gauss_smooth=1,interpolate=False)
				self._bkg_round_3()
				self.flux -= self.bkg
				if self.corr_correction:
					if self.verbose > 0:
						print('Background correlation correction')
					self.correlation_corrector()


			if self.calibrate:
				print('Field calibration')
				self.field_calibrate()

			
			self.lc, self.sky = self.diff_lc(plot=True,diff=self.diff,tar_ap=tar_ap,sky_in=sky_in,sky_out=sky_out)
		except Exception:
			print(traceback.format_exc())

		
		

	def make_lc(self,aperture = None,bin_size=0,zeropoint=None,scale='counts',clip = False):
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

	def lc_events(self,lc = None,err=None,duration=10,sig=5):
		"""
		Use clustering to detect individual high SNR events in a light curve.
		Clustering isn't incredibly robust, so it could be better.

		-----------------
		Inputs (optional)
		-----------------
		err : array
			flux error to be used in weighting of events
		duration : int 
			How long an event needs to last for before being detected
		sig : float
			significance of the detection above the background
		--------
		Returns
		-------
		self.events : list
			list of light curves for all identified events 
		"""
		if lc is None:
			lc = deepcopy(self.lc)
		if lc.shape[0] > lc.shape[1]:
			lc = lc.T
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

	def event_plotter(self,**kwargs):
		"""
		Lazy plotting tool for checking the detected events.
		"""
		if self.events is None:
			self.lc_events(**kwargs)
		plt.figure()
		plt.plot(self.lc[0],self.lc[1],'k.')
		for i in range(len(self.events)):
			plt.plot(self.events[i,0],self.events[i,1],'*',label='Event {}'.format(i))
		plt.xlabel('MJD')
		plt.ylabel('Flux')


	def detrend_transient(self,lc=None,err=None,Mask=None,variable=False,sig = 5, 
						  sig_up = 3, sig_low = 10, tail_length='auto',plot=False):
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
		size = int(lc.shape[1] * 0.01)
		if size % 2 == 0: 
			size += 1
		for i in range(len(break_inds)-1):
			section = lc[:,break_inds[i]:break_inds[i+1]]

			mask_section = masked[:,break_inds[i]:break_inds[i+1]]
			if np.nansum(mask_section) < 10:
				mask_section[1,:] = np.nanmedian(masked[1,:])
				if np.nansum(abs(mask_section)) < 10:
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
			
		if plot:
			plt.figure()
			plt.plot(self.lc[0],self.lc[1])
			plt.plot(self.lc[0,nonan],trends,'.')
		detrend = deepcopy(self.lc)
		detrend[1,nonan] -= trends
		return detrend

	def detrend_stellar_var(self,lc=None,err=None,Mask=None,variable=False,sig = None, 
							sig_up = 5, sig_low = 10, tail_length=''):
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
		self._midsector_break = break_inds
		#lc[Mask] = np.nan
		
		if variable:
			size = int(lc.shape[1] * 0.08)
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


	def bin_interp(self,lc=None,time_bin=6/26):
		if lc is None:
			lc = self.lc
		if lc.shape[0] > lc.shape[1]:
			lc = lc.T
		binned = self.bin_data(lc=lc,time_bin=time_bin)
		finite = np.isfinite(binned[1])
		f1 = interp1d(binned[0,finite], binned[1,finite], kind='linear',fill_value='extrapolate')
		smooth = f1(lc[0])
		return smooth


	def detrend_star(self,lc=None):
		if lc is None:
			lc = self.lc
		if lc.shape[0] > lc.shape[1]:
			lc = lc.T
		# clip outliers with grads 
		raw_flux = lc[1]
		rad = grads_rad(raw_flux)
		ind = (rad > np.nanmedian(rad)+5*np.nanstd(rad))
		flux = deepcopy(raw_flux)
		flux[ind] = np.nan
		smooth = self.bin_interp(lc = np.array([lc[0],flux]))
		
		sub = flux - smooth
		
		rad = grad_flux_rad(sub)
		ind = rad > np.nanmedian(rad)+2*np.nanstd(rad)
		
		mask = ind * 1
		mask = convolve(mask,np.ones((3))) > 0
		
		temp = deepcopy(lc)
		temp[1,mask] = np.nan

		size = int(lc.shape[1] * 0.05)
		if size % 2 == 0: size += 1
		finite = np.isfinite(temp[1])
		smooth = savgol_filter(temp[1,finite],size,2)
		f1 = interp1d(temp[0,finite], smooth, kind='linear',fill_value='extrapolate')
		smooth = f1(temp[0])
		
		detrended = deepcopy(lc)
		detrended[1] -= smooth
		return detrended


	### serious calibration 
	def isolated_star_lcs(self):
		if self.dec < -30:
			if self.verbose > 0:
				print('Target is below -30 dec, calibrating to SkyMapper photometry.')
			table = Get_Catalogue(self.tpf,Catalog='skymapper')
			table = Skymapper_df(table)
			system = 'skymapper'
		else:
			if self.verbose > 0:
				print('Target is above -30 dec, calibrating to PS1 photometry.')
			table = Get_Catalogue(self.tpf,Catalog='ps1')
			system = 'ps1'

		if self.diff:
			tflux = self.flux + self.ref
		else:
			tflux = self.flux
			

		ind = (table.imag.values < 19) & (table.imag.values > 14)
		tab = table.iloc[ind]
		x,y = self.wcs.all_world2pix(tab.RAJ2000.values,tab.DEJ2000.values,0)
		tab['col'] = x
		tab['row'] = y
		
		e, dat = Tonry_reduce(tab,system=system)
		self.ebv = e[0]

		gr = (dat.gmag - dat.rmag).values
		ind = (gr < 1) & (dat.imag.values < 17)
		d = dat.iloc[ind]
		x,y = self.wcs.all_world2pix(d.RAJ2000.values,d.DEJ2000.values,0)
		d['col'] = x
		d['row'] = y
		pos_ind = (1 < x) & (x < self.ref.shape[0]-2) & (1 < y) & (y < self.ref.shape[0]-2)
		d = d.iloc[pos_ind]

		# account for crowding 
		for i in range(len(d)):
			x = d.col.values[i]
			y = d.row.values[i]
			
			dist = np.sqrt((tab.col.values-x)**2 + (tab.row.values-y)**2)
			
			ind = dist < 1.5
			close = tab.iloc[ind]
			
			d['gmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.gmag.values,25))) + 25
			d['rmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.rmag.values,25))) + 25
			d['imag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.imag.values,25))) + 25
			d['zmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.zmag.values,25))) + 25
			if system == 'ps1':
				d['ymag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.ymag.values,25))) + 25
		# convert to tess mags
		if len(d) < 10:
			print('!!!WARNING!!! field calibration is unreliable, using the default zp = 20.44')
			self.zp = 20.44
			self.zp_e = 0.5
			# backup for when messing around with flux later
			self.tzp = 20.44
			self.tzp_e = 0.5
			return
		if system == 'ps1':
			d = PS1_to_TESS_mag(d,ebv=self.ebv)
		else:
			d = SM_to_TESS_mag(d,ebv=self.ebv)

		
		flux = []
		eflux = []
		eind = np.zeros(len(d))
		for i in range(len(d)):
			#if self.phot_method == 'aperture':
			mask = np.zeros_like(self.ref)
			mask[int(d.row.values[i] + .5),int(d.col.values[i] + .5)] = 1
			mask = convolve(mask,np.ones((3,3)))
			flux += [np.nansum(tflux*mask,axis=(1,2))]
			m2 = np.zeros_like(self.ref)
			m2[int(d.row.values[i] + .5),int(d.col.values[i] + .5)] = 1
			m2 = convolve(m2,np.ones((7,7))) - convolve(m2,np.ones((5,5)))
			eflux += [np.nansum(tflux*m2,axis=(1,2))]
			mag = -2.5*np.log10(np.nansum((self.ref*m2))) + 20.44
			#elif self.phot_method == 'psf':
			#	self.psf_photometry(xPix=d.col.values[i],yPix=d.row.values[i],snap=None,diff=False)
			
			if (mag <= d.tmag.values[i]+1):# | (mag <= 17):
				eind[i] = 1
		eind = eind == 0
		flux = np.array(flux)
		eflux = np.array(eflux)
		#eind = abs(eflux) > 20
		flux[~eind] = np.nan

		return flux[eind], d.iloc[eind]


	def field_calibrate(self,zp_single=True,plot=None,savename=None):
		"""
		In-situ flux calibration for TESSreduce light curves. This uses the
		flux calibration method developed in Ridden-Harper et al. 2021 where a broadband 
		filter is reconstructed by a linear combination of PS1 filters + a non linear colour term.
		Here, we calibrate to all PS1 stars in the tpf region by first calculating the 
		stellar extinction in E(B-V) using stellar locus regression. We then identify all reasonably 
		isolated stars with g-r < 1 and i < 17 in the TPF. For each isolated source we calculate the
		expected TESS magnitude, including all sources within 2.5 pixels (52.5''), and compare 
		that to TESS aperture photometry. Averaging together all valid sources gives us a 
		good representation of the TESS zeropoint. 

		Since we currently only use PS1 photometry, this method is only avaiable in areas of 
		PS1 coverage, so dec > -30. 

		-------
		Options
		-------
		zp_single : bool
			if True all points through time are averaged to a single zp
			if False then the zp is time varying, creating an extra photometric correction
			for light curves, but with increased error in the zp.
		plot : bool
			if True then diagnostic plots will be created
		-------
		Returns
		-------
		self.ebv : float 
			estimated E(B-V) extinction from stellar locus regression
		self.zp/tzp : float
			TESS photometric zeropoint
		self.zp_e/tzp_e : float
			error in the photometric zeropoint
		"""
		if plot is None:
			plot = self.plot
		if savename is None:
			savename = self.savename
		if self.dec < -30:
			if self.verbose > 0:
				print('Target is below -30 dec, calibrating to SkyMapper photometry.')
			table = Get_Catalogue(self.tpf,Catalog='skymapper')
			table = Skymapper_df(table)
			system = 'skymapper'
		else:
			if self.verbose > 0:
				print('Target is above -30 dec, calibrating to PS1 photometry.')
			table = Get_Catalogue(self.tpf,Catalog='ps1')
			system = 'ps1'
		x,y = self.wcs.all_world2pix(table.RAJ2000.values,table.DEJ2000.values,0)
		table['col'] = x
		table['row'] = y
		self.cat = table
		
		ref = deepcopy(self.ref)
		m = ((self.mask & 1 == 0) & (self.mask & 2 == 0) ) * 1.
		m[m==0] = np.nan
		ref_bkg = np.nanmedian(ref * m)
		
		ref -= ref_bkg
		if self.diff:
			tflux = self.flux + ref
		else:
			tflux = self.flux
			

		ind = (table.imag.values < 19) & (table.imag.values > 14)
		tab = table.iloc[ind]
		
		e, dat = Tonry_reduce(tab,plot=plot,savename=savename,system=system)
		self.ebv = e[0]

		gr = (dat.gmag - dat.rmag).values
		ind = (gr < 1) & (dat.imag.values < 17)
		d = dat.iloc[ind]
		
		x,y = self.wcs.all_world2pix(d.RAJ2000.values,d.DEJ2000.values,0)
		d['col'] = x
		d['row'] = y
		pos_ind = (3 < x) & (x < self.ref.shape[1]-3) & (3 < y) & (y < self.ref.shape[0]-3)
		d = d.iloc[pos_ind]
		

		# account for crowding 
		for i in range(len(d)):
			x = d.col.values[i]
			y = d.row.values[i]
			
			dist = np.sqrt((tab.col.values-x)**2 + (tab.row.values-y)**2)
			
			ind = dist < 1.5
			close = tab.iloc[ind]
			
			d['gmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.gmag.values,25))) + 25
			d['rmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.rmag.values,25))) + 25
			d['imag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.imag.values,25))) + 25
			d['zmag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.zmag.values,25))) + 25
			if system == 'ps1':
				d['ymag'].iloc[i] = -2.5*np.log10(np.nansum(mag2flux(close.ymag.values,25))) + 25
		# convert to tess mags
		if len(d) < 10:
			print('!!!WARNING!!! field calibration is unreliable, using the default zp = 20.44')
			self.zp = 20.44
			self.zp_e = 0.05
			# backup for when messing around with flux later
			self.tzp = 20.44
			self.tzp_e = 0.05
			return
		if system == 'ps1':
			d = PS1_to_TESS_mag(d,ebv=self.ebv)
		else:
			d = SM_to_TESS_mag(d,ebv=self.ebv)

		
		flux = []
		eflux = []
		eind = np.zeros(len(d))
		for i in range(len(d)):
			mask = np.zeros_like(self.ref)
			xx = int(d.col.values[i] + .5); yy = int(d.row.values[i] + .5)
			mask[yy,xx] = 1
			mask = convolve(mask,np.ones((3,3)))
			if self.phot_method == 'aperture':
				flux += [np.nansum(tflux*mask,axis=(1,2))]
			elif self.phot_method == 'psf':
				flux += [self.psf_photometry(xPix=xx,yPix=yy,snap='ref',diff=False)]
			m2 = np.zeros_like(self.ref)
			m2[int(d.row.values[i] + .5),int(d.col.values[i] + .5)] = 1
			m2 = convolve(m2,np.ones((7,7))) - convolve(m2,np.ones((5,5)))
			eflux += [np.nansum(tflux*m2,axis=(1,2))]
			mag = -2.5*np.log10(np.nansum((ref*m2))) + 20.44

			
			if (mag <= d.tmag.values[i]+1):# | (mag <= 17):
				eind[i] = 1
		eind = eind == 0

		flux = np.array(flux)
		eflux = np.array(eflux)
		#eind = abs(eflux) > 20
		if self.phot_method == 'aperture':
			flux[~eind] = np.nan
		

		#calculate the zeropoint
		zp = d.tmag.values[:,np.newaxis] + 2.5*np.log10(flux) 
		if len(zp) == 0:
			zp = np.array([20.44])
		
		mzp = np.zeros_like(zp[0]) * np.nan
		stdzp = np.zeros_like(zp[0]) * np.nan
		for i in range(zp.shape[1]):
			#averager = calcaverageclass()
			mean, med, std = sigma_clipped_stats(zp[eind,i], sigma=3.0)
			#averager.calcaverage_sigmacutloop(zp[eind,i])
			mzp[i] = med#averager.mean
			stdzp[i] = std#averager.stdev

		#averager = calcaverageclass()
		mean, med, std = sigma_clipped_stats(mzp[np.isfinite(mzp)], sigma=3.0)
		#averager.calcaverage_sigmacutloop(mzp[np.isfinite(mzp)],noise=stdzp[np.isfinite(mzp)])

		if plot:
			plt.figure()
			nonan = np.isfinite(self.ref)
			plt.imshow(ref,origin='lower',vmax = np.percentile(ref[nonan],80),vmin=np.percentile(ref[nonan],10))
			plt.scatter(d.col.iloc[eind],d.row.iloc[eind],color='r')
			plt.title('Calibration sources')
			plt.ylabel('Row',fontsize=15)
			plt.xlabel('Column',fontsize=15)
			plt.colorbar()
			plt.show()
			if savename is not None:
				plt.savefig(savename + 'cal_sources.pdf', bbox_inches = "tight")


			mask = sigma_mask(mzp,3)
			plt.figure(figsize=(3*fig_width,1*fig_width))
			plt.subplot(121)
			plt.hist(mzp[mask],alpha=0.5)
			#plt.axvline(averager.mean,color='C1')
			#plt.axvspan(averager.mean-averager.stdev,averager.mean+averager.stdev,alpha=0.3,color='C1')
			#plt.axvspan(med-std,med+std,alpha=0.3,color='C1')
			med = med
			low = med-std
			high = med+std
			plt.axvline(med,ls='--',color='k')
			plt.axvline(low,ls=':',color='k')
			plt.axvline(high,ls=':',color='k')

			s = '$'+str((np.round(med,3)))+'^{+' + str((np.round(high-med,3)))+'}_{'+str((np.round(low-med,3)))+'}$'
			plt.annotate(s,(.70,.8),fontsize=13,xycoords='axes fraction')
			plt.xlabel('Zeropoint',fontsize=15)
			plt.ylabel('Occurrence',fontsize=15)
			plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))

			plt.subplot(122)
			plt.plot(self.tpf.time.mjd[mask],mzp[mask],'.',alpha=0.5)
			#plt.axhspan(averager.mean-averager.stdev,averager.mean+averager.stdev,alpha=0.3,color='C1')
			#plt.axhline(averager.mean,color='C1')
			#plt.axhspan(med-std,med+std,alpha=0.3,color='C1')

			plt.axhline(low,color='k',ls=':')
			plt.axhline(high,color='k',ls=':')
			plt.axhline(med,color='k',ls='--')

			plt.ylabel('Zeropoint',fontsize=15)
			plt.xlabel('MJD',fontsize=15)
			plt.tight_layout()
			plt.show()
			if savename is not None:
				plt.savefig(savename + 'cal_zp.pdf', bbox_inches = "tight")

		if zp_single:
			mzp = med#averager.mean
			stdzp = std#averager.stdev
			compare = abs(mzp-20.44) > 2

		else:
			zp = np.nanmedian(zp,axis=0)
			mzp,stdzp = smooth_zp(zp, self.tpf.time.mjd)
			compare = (abs(mzp-20.44) > 2).any()

		if compare:
			print('!!!WARNING!!! field calibration is unreliable, using the default zp = 20.44')
			self.zp = 20.44
			self.zp_e = 0.5
			# backup for when messing around with flux later
			self.tzp = 20.44
			self.tzp_e = 0.5
		else:
			self.zp = mzp
			self.zp_e = stdzp
			# backup for when messing around with flux later
			self.tzp = mzp
			self.tzp_e = stdzp

		return

	def to_mag(self,zp=None,zp_e=0):
		"""
		Convert the TESS lc into magnitude space.
		This is non reversible, since negative values will be lost.

		Inputs:
			zp: zeropoint to use for conversion. If None, use the default zp from the object.
			zp_e: error on the zeropoint to use for conversion. If None, use the default zp_e from the object.
		
		Outputs:
			lc: lightcurve in magnitude space.
		"""
		if (zp is None) & (self.zp is not None):
			zp = self.zp
			zp_e = self.zp_e
		elif (zp is None) & (self.zp is None):
			self.field_calibrate()
			zp = self.zp
			zp_e = self.zp_e

		mag = -2.5*np.log10(self.lc[1]) + zp
		mag_e = np.sqrt((2.5/np.log(10) * self.lc[2]/self.lc[1])**2 + zp_e**2)

		lc = deepcopy(self.lc)
		lc[1] = mag
		lc[2] = mag_e

		#self.lc[1] = mag
		#self.lc[2] = mag_e
		#self.lc_units = 'AB mag'
		return lc

	def to_flux(self,zp=None,zp_e=0,flux_type='mjy',plot=False):
		"""
		Convert the TESS lc to physical flux. Either the field calibrated zp 
		or a given zp can be used. 

		-----------------
		Inputs (optional)
		-----------------
		zp : float
			tess zeropoint 
		zp_e : float
			error in the tess zeropoint
		flux_type : str
			Valid options:
			mjy 
			jy
			erg/cgs
			tess/counts
		-------
		Options
		-------
		plot : bool
			plot the field calibration figures, if used.

		-------
		Returns
		-------
		self.lc : array
			converted to the requested unit
		self.zp : float
			updated with the new zeropoint 
		self.zp_e : float 
			updated with the new zeropoint error
		self.lc_units : str
			updated with the flux unit used

		"""
		if (zp is None) & (self.zp is not None):
			zp = self.zp
			zp_e = self.zp_e
		elif (zp is None) & (self.zp is None):
			print('Calculating field star zeropoint')
			self.field_calibrate()
			zp = self.zp
			zp_e = self.zp_e

		if flux_type.lower() == 'mjy':
			flux_zp = 16.4
		elif flux_type.lower() == 'jy':
			flux_zp = 8.9
		elif (flux_type.lower() == 'erg') | (flux_type.lower() == 'cgs'):
			flux_zp = -48.6
		elif (flux_type.lower() == 'tess') | (flux_type.lower() == 'counts'):
			if self.tzp is None:
				print('Calculating field star zeropoint')
				self.field_calibrate(plot=plot)
			flux_zp = self.tzp

		else:
			m = '"'+flux_type + '" is not a valid option, please choose from:\njy\nmjy\ncgs/erg\ntess/counts'
			raise ValueError(m)

		flux = self.lc[1] * 10**((zp - flux_zp)/-2.5)
		flux_e2 = ((10**((zp-flux_zp)/-2.5))**2 * self.lc[2]**2 + 
					(self.lc[1]/-2.5 * 10**((zp-flux_zp)/-2.5))**2 * zp_e**2)
		flux_e = np.sqrt(flux_e2)
		self.lc[1] = flux
		self.lc[2] = flux_e


		if flux_type.lower() == 'mjy':
			self.zp = self.zp * 0 + 16.4
			self.zp_e = 0
			self.lc_units = 'mJy'
		if flux_type.lower() == 'jy':
			self.zp = self.zp * 0 + 8.9
			self.zp_e = 0
			self.lc_units = 'Jy'
		elif (flux_type.lower() == 'erg') | (flux_type.lower() == 'cgs'):
			self.zp = self.zp * 0 -48.6
			self.zp_e = 0
			self.lc_units = 'cgs'
		elif (flux_type.lower() == 'tess') | (flux_type.lower() == 'counts'):
			self.zp = self.tzp
			self.zp_e = 0
			self.lc_units = 'Counts'
		return 
		
		




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

def external_save_cat(radec,size,cutCornerPx,image_path,save_path,maglim):
	
	file = _Extract_fits(image_path)
	wcsItem = wcs.WCS(file[1].header)
	file.close()
	
	ra = radec[0]
	dec = radec[1]

	gp,gm, source = Get_Gaia_External(ra,dec,cutCornerPx,size,wcsItem,magnitude_limit=maglim)
	gaia  = pd.DataFrame(np.array([gp[:,0],gp[:,1],gm,source]).T,columns=['ra','dec','mag','Source'])

	gaia.to_csv(f'{save_path}/local_gaia_cat.csv',index=False)

def _load_external_cat(path,maglim):

	gaia = pd.read_csv(f'{path}/local_gaia_cat.csv')
	gaia = gaia[gaia['mag']<(maglim-0.5)]
	gaia = gaia[['ra','dec','mag']]
	return gaia

### Serious source mask

def Cat_mask(tpf,cataloge_path=None,maglim=19,scale=1,strapsize=3,badpix=None,ref=None,sigma=3):
	"""
	Make a source mask from the PS1 and Gaia catalogs.

	------
	Inputs
	------
	tpf : lightkurve target pixel file
		tpf of the desired region
	maglim : float
		magnitude limit in PS1 i band  and Gaia G band for sources.
	scale : float
		scale factor for default mask size 
	strapsize : int
		size of the mask for TESS straps 
	badpix : str
		not implemented correctly, so just ignore! 

	-------
	Returns
	-------
	total mask : bitmask
		a bitwise mask for the given tpf. Bits are as follows:
		0 - background
		1 - catalogue source
		2 - saturated source
		4 - strap mask
		8 - bad pixel (not used)
	"""
	
	wcs = tpf.wcs
	image = tpf.flux[100]
	image = strip_units(image)
	gp,gm = Get_Gaia(tpf,magnitude_limit=maglim)
	gaia  = pd.DataFrame(np.array([gp[:,0],gp[:,1],gm]).T,columns=['x','y','mag'])
	#gaia  = _load_external_cat(cataloge_path,maglim)
	#coords = tpf.wcs.all_world2pix(gaia['ra'],gaia['dec'], 0)
	#gaia['x'] = coords[0]
	#gaia['y'] = coords[1]

	#if tpf.dec > -30:
	#	pp,pm = Get_PS1(tpf,magnitude_limit=maglim)
	#	ps1   = pd.DataFrame(np.array([pp[:,0],pp[:,1],pm]).T,columns=['x','y','mag'])
	#	mp  = ps1_auto_mask(ps1,image,scale)
	#else:
	#	mp = {}
	#	mp['all'] = np.zeros_like(image)
	sat = Big_sat(gaia,image,scale)
	if ref is None:
		mg  = gaia_auto_mask(gaia,image,scale)
		mask = (mg['all'] > 0).astype(int) * 1 # assign 1 bit
	else:
		mg = np.zeros_like(ref,dtype=int)
		mean, med, std = sigma_clipped_stats(ref)
		lim = med + sigma * std
		ind = ref > lim
		mg[ind] = 1
		mask = (mg > 0).astype(int) * 1 # assign 1 bit
	

	sat = (np.nansum(sat,axis=0) > 0).astype(int) * 2 # assign 2 bit 
	#mask = ((mg['all']+mp['all']) > 0).astype(int) * 1 # assign 1 bit
	
	if strapsize > 0: 
		strap = Strap_mask(image,tpf.column,strapsize).astype(int) * 4 # assign 4 bit 
	else:
		strap = np.zeros_like(image,dtype=int)
	if badpix is not None:
		bp = cat_mask.Make_bad_pixel_mask(badpix, file)
		totalmask = mask | sat | strap | bp
	else:
		totalmask = mask | sat | strap
	
	return totalmask, gaia

	


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
