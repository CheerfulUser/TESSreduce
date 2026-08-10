"""
Import packages!
"""
import traceback
import os
import time
import pandas as pd
import numpy as np

import lightkurve as lk

from copy import deepcopy

from scipy.ndimage import convolve
from scipy.ndimage import shift

from scipy.signal import savgol_filter

from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip

import multiprocessing
from joblib import Parallel, delayed

def _available_cores(verbose=1):
	"""Return the number of CPU cores available to this process.

	Checks in priority order:
	  1. SLURM_CPUS_PER_TASK - cores allocated by SLURM scheduler
	  2. os.sched_getaffinity - respects cgroup/affinity limits (Linux)
	  3. multiprocessing.cpu_count - total node CPUs (fallback)
	"""
	slurm = os.environ.get('SLURM_CPUS_PER_TASK')
	if slurm is not None:
		try:
			n = int(slurm)
			if verbose > 0:
				print(f'[tessreduce] _available_cores: SLURM_CPUS_PER_TASK={slurm} → {n} cores')
			return n
		except ValueError:
			pass
	try:
		n = len(os.sched_getaffinity(0))
		if verbose > 0:
			print(f'[tessreduce] _available_cores: sched_getaffinity → {n} cores')
		return n
	except AttributeError:
		pass
	n = multiprocessing.cpu_count()
	if verbose > 0:
		print(f'[tessreduce] _available_cores: cpu_count fallback → {n} cores')
	return n

from .catalog_tools import *
from .calibration_tools import *
from .ground_tools import ground
from .lastpercent import *
from .helpers import *
from .cat_mask import Cat_mask

# turn off runtime warnings (lots from logic on nans)
import warnings
# nuke warnings because sigma clip is extremely annoying 
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning) 
_visible_dep_warning = getattr(getattr(np, 'exceptions', np), 'VisibleDeprecationWarning', None)
if _visible_dep_warning is not None:
	warnings.filterwarnings("ignore", category=_visible_dep_warning)
pd.options.mode.chained_assignment = None

# set the package directory so we can load in a file later
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

def _fit_bkg_surface_frame(residual, exclude_mask, box_size, filter_size, sigma):
	from photutils.background import Background2D, MedianBackground
	from astropy.stats import SigmaClip
	sc = SigmaClip(sigma=sigma, maxiters=5)
	finite_vals = residual[~exclude_mask & np.isfinite(residual)]
	med = np.nanmedian(finite_vals)
	std = np.nanstd(finite_vals)
	transient_mask = exclude_mask | (residual > med + 5 * std)
	try:
		b = Background2D(residual, box_size=box_size, filter_size=filter_size,
						 sigma_clip=sc, bkg_estimator=MedianBackground(),
						 mask=transient_mask, fill_value=0.0)
		return b.background
	except Exception:
		return np.full_like(residual, np.nanmedian(residual[~transient_mask]))

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27				# Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0		 # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def _subtract_residual_surface(bkg, flux, bkgmask, box_size=20, filter_size=5, sigma=3.0, backend='loky', verbose=0):
	"""
	Fit and subtract a smooth 2D residual surface per frame using photutils
	Background2D. Operates on (flux - bkg) to capture large-scale structure
	missed by the primary background estimation.

	Parameters
	----------
	bkg : np.ndarray (T, X, Y)
		Current background estimate; updated in-place.
	flux : np.ndarray (T, X, Y)
		Raw flux cube.
	bkgmask : np.ndarray (X, Y) or (T, X, Y)
		Background mask as produced by the background function: NaN where
		pixels are excluded (sources/straps), 1.0 where valid background.
		If 3D, a pixel is excluded if it is NaN in any frame.
	box_size : int
		Side length of the background mesh boxes in pixels.
	filter_size : int
		Size of the median filter applied to the mesh before interpolation.
	sigma : float
		Sigma threshold for iterative sigma-clipping within each box.

	Returns
	-------
	bkg : np.ndarray (T, X, Y)
		Background cube with the per-frame 2D surface added.
	"""
	from photutils.background import Background2D, MedianBackground
	from astropy.stats import SigmaClip
	from joblib import Parallel, delayed

	bkgmask = np.asarray(bkgmask)
	if bkgmask.ndim == 3:
		exclude_mask = np.any(np.isnan(bkgmask), axis=0)
	else:
		exclude_mask = np.isnan(bkgmask)

	n_jobs = _available_cores()
	residuals = flux - bkg
	corrections = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
		delayed(_fit_bkg_surface_frame)(residuals[i], exclude_mask, box_size, filter_size, sigma)
		for i in range(flux.shape[0]))
	bkg += np.array(corrections)

	return bkg


class tessreduce():

	def __init__(self,ra=None,dec=None,name=None,obs_list=None,tpf=None,size=90,sector=None,
			  	 flux=None,mjd=None,wcs=None,camera=None,ccd=None,shifts=None,
				 reduce=True,align=True,diff=True,corr_correction=False,kernel_match=False,calibrate=True,sourcehunt=True,
				 phot_method='aperture',imaging=False,parallel=True,num_cores=-1,backend='loky',diagnostic_plot=False,plot=True,
				 savename=None,quality_bitmask='hard',cache_dir=None,cache=True,catalogue_path=False,
				 shift_method='sep_core',use_error_image=False,prf_path=None,verbose=1,col_offset=0,
				 bkg_temporal_window=501,ref_ind=None,ref_type='stack',ref_time_window=2,vector_path=None,
				 smooth_motion=False,orbit_ref=False,bkg_gauss_sigma=4,create_lc=True,center_mask=True,timing=False):

		"""
		Class for extracting reduced TESS photometry around a target coordinate or event. 

		Parameters
		----------
		ra : float
			Right ascension of the target object. The default is None.
		dec : float
			Declination of the target object. The default is None.
		name : str
			Name of the object, used in saving. The default is None.
		obs_list : array_like, optional
			Array generated by the sn_lookup and spacetime_lookup functions. The default is None.
		tpf : target pixel file, optional
			TESS target pixel file. The default is None.
		size : int, optional
			Size in pixels of the cutout, larger cutout sizes give better background subtractions. The default is 90.
		sector : int, optional
			Sector of observations. The default is None.
		phot_method : str, optional
			Select the photometry method used in the reduction. Choose between 'aperture' and 'PSF'. The default is 'aperture'.

		Options
		-------
		reduce : bool, optional
			Perform photometric reduction processes for the target region. The default is True.
		align : bool, optional
			Shift images to align stars with a reference frame. The default is True.
		diff : bool, optional
			Calculate difference imaging between each frame and a reference frame. The default is True.
		corr_correction : bool, optional
			Final correction step that operates on pixels which have high correlation between their lightcurve and the background. The default is True.
		calibrate : bool, optional
			Performs photometric calibration on the datacube using PS1 or SkyMapper data. The default is True.
		sourcehunt : bool, optional
			Searches for sources in the background and masks them. Prevents asteroids and other transients from being included in the background. The default is True.
		imaging : bool, optional
			Retrieves PanSTARRS or SkyMapper photometry of the region. The default is False.
		parallel : bool, optional
			Perform computation with parallel processing using 'num_cores'. The default is True.
		num_cores : int, optional
			Number of cores to run parallel process on. The default is -1 which uses max system cores.
		backend : str, optional
			joblib backend used for parallel processing. 'loky' (default) is robust in interactive
			sessions such as Jupyter/IPython. Use 'multiprocessing' on servers/clusters (e.g. SLURM/OzStar)
			where it is required.
		diagnostic_plot : bool, optional
			During reduction, plot figures which outline various calculation steps, such as the image shifts over time or the zeropoint calculation. The default is False.
		plot : bool, optional
			During reduction, plot resulting light curve. The default is True.
		savename : str, optional
			Save name for the outputs. The default is None.
		quality_bitmask : str, optional
			Parameter for lightkurve download of TESS TPF. The default is 'default'.
		cache_dir : str, optional
			Directory to cache files. The default is None.
		catalogue_path : str, optional
			Path to required catalogs for when using TESSreduce in offline mode. The default is False.
		prf_path : str, optional
			Path to local TESS PRF files. The default is currently a specific location on the OzStar supercomputer.
		verbose : int, optional
			Controls the level of verbosity. 0 is silent, 1 (default) prints reduction stage
			announcements, 2 additionally prints joblib's per-task Parallel output.
		timing : bool, optional
			Print execution time reports for major pipeline blocks in background() and reduce(). The default is False.

		"""

		# Field Specific
		self.ra = ra
		self.dec = dec 
		self.name = name
		self.size = size
		self.sector = sector
		self.tpf = tpf

		# Reduction Process Specific
		self._create_lc = create_lc
		self.align = align
		self.calibrate = calibrate
		self.corr_correction = corr_correction
		self.diff = diff
		self.kernel_match = kernel_match
		self.orbit_ref = orbit_ref
		self._bkg_gauss_sigma = bkg_gauss_sigma
		self._center_mask = center_mask
		self.imaging = imaging
		self.parallel = parallel
		self.backend = backend
		self._col_offset = col_offset
		if num_cores == -1 or isinstance(num_cores, str):
			self.num_cores = _available_cores(verbose=verbose)
		else:
			self.num_cores = num_cores
		self._assign_phot_method(phot_method)
		self._sourcehunt = sourcehunt
		self.verbose = verbose
		self._shift_method = shift_method
		self._use_error_image = use_error_image
		self._bkg_temporal_window = bkg_temporal_window
		self._force_ref_ind = ref_ind
		self._ref_type = ref_type
		self._ref_time_window = ref_time_window
		self._quality_bitmask = quality_bitmask
		self._smooth_motion = smooth_motion
		self._timing = timing
		self._cache_path = None

		# SLURM environment diagnostics
		if verbose > 0:
			_slurm_vars = ['SLURM_CPUS_PER_TASK', 'SLURM_NTASKS', 'SLURM_NTASKS_PER_NODE',
						   'SLURM_JOB_CPUS_PER_NODE', 'SLURM_CPUS_ON_NODE']
			_slurm_env = {k: os.environ.get(k, 'not set') for k in _slurm_vars}
			print(f'[tessreduce] SLURM env: {_slurm_env}')
			print(f'[tessreduce] num_cores resolved to: {self.num_cores}')

		# Offline Paths 
		if catalogue_path is None:
			catalogue_path = os.getcwd()
		elif catalogue_path is False:
			catalogue_path = None
		self._catalogue_path = catalogue_path
		self.imaging = imaging
		self._prf_path = prf_path
		self._vector_path = vector_path

		# Plotting
		self.plot = plot
		self.diagnostic_plot = diagnostic_plot
		self.savename = savename

		# Optionally given
		self.flux = flux
		self.mjd = mjd
		self.wcs = wcs
		self.camera = camera
		self.ccd = ccd 
		self.shift = shifts

		# Calculated 
		self.mask = None
		#self._over_sub = None
		self.bkg = None
		# self.flux = None
		self.delta_kernel = None
		self.ref = None
		self.ref_ind = None
		self.qe = None
		self.lc = None
		self.sky = None
		self.events = None
		self.zp = None
		self.zp_e = None
		self.sn_name = None
		self.ebv = 0
		self.epsf = None
		# repeat for backup
		self.tzp = None
		self.tzp_e = None
		
		# light curve units 
		self.lc_units = 'Counts'

		# Generate coordinate information from 'obs_list'
		if obs_list is not None:
			if isinstance(obs_list,list):
				obs_list = np.array(obs_list,dtype=object)
				if len(obs_list.shape) > 1:
					obs_list = obs_list[obs_list[:,3].astype('bool')][0]
				self.ra = obs_list[0]
				self.dec = obs_list[1]
				self.sector = obs_list[2]
			elif isinstance(obs_list, pd.DataFrame):
				self.ra = obs_list['RA'].to_numpy()[0]
				self.dec = obs_list['DEC'].to_numpy()[0]
				self.sector = obs_list['Sector'].to_numpy()

		# Generate coordinate information from 'tpf'
		if tpf is not None:
			if isinstance(tpf, str):
				self.tpf = lk.TessTargetPixelFile(tpf,quality_bitmask=self._quality_bitmask)
			self.flux = strip_units(self.tpf.flux)
			if self._use_error_image:
				self.eflux = strip_units(self.tpf.flux_err)
			else:
				self.eflux = None
			self.flux[np.isnan(self.flux)] = 0
			self.mjd = self.tpf.time.mjd
			self.wcs = self.tpf.wcs
			self.ra = self.tpf.ra
			self.dec = self.tpf.dec
			self.size = self.tpf.flux.shape[1]
			if self.tpf.sector is not None:
				self.sector = self.tpf.sector
			if not self.sector:
				# Try FITS header directly (tessellate files leave SECTOR blank)
				try:
					s = self.tpf.hdu[0].header.get('SECTOR')
					if s:
						self.sector = int(s)
				except Exception:
					pass
			if not self.sector:
				# Parse from filename: sector27_... or s0027...
				import re
				m = re.search(r'sector(\d+)', str(tpf), re.IGNORECASE)
				if not m:
					m = re.search(r's(\d{4})', str(tpf))
				if m:
					self.sector = int(m.group(1))
			if not self.sector:
				self.sector = 999

			self.column = self.tpf.column
			self.row = self.tpf.row
			self.camera = self.tpf.camera
			self.ccd = self.tpf.ccd

		# -- Allow for cube to be given directly, but require MJD, WCS, Sector to be given as well -- #
		elif self.flux is not None:
			if self.mjd is None:	# obviously need the time
				m = 'If flux is given, the MJD must also be given.'
				raise ValueError(m)
			if self.wcs is None:	# obviously need the WCS
				m = 'If flux is given, the WCS must also be given.'
				raise ValueError(m)
			if self.sector is None:	# obviously need the sector
				m = 'If flux is given, the sector must also be given.'
				raise ValueError(m)
			if self._force_ref_ind is None:		# this is because we dont have the quality flags anymore, so can't get a good ref easily
				m = 'If flux is given, the reference frame index must also be given.'
				raise ValueError(m)
			if self.camera is None:
				m = 'If flux is given, the camera must also be given.'
				raise ValueError(m)
			if self.ccd is None:
				m = 'If flux is given, the CCD must also be given.'
				raise ValueError(m)
			
			self.size = self.flux.shape[1]
			self.ra,self.dec = self.wcs.all_pix2world(self.size//2,self.size//2,0)
			self.eflux = None
			self.column = 0
			self.row = 0
			self.rawflux = deepcopy(self.flux)


		# Retrieve TPF
		elif self.check_coord():
			if self.verbose>0:
				print('Downloading TPF from TESScut')
			self.get_TESS(quality_bitmask=self._quality_bitmask,cache_dir=cache_dir,cache=cache)
			self._get_gaia()

		self.ground = ground(ra = self.ra, dec = self.dec)

		if reduce:
			self.reduce()

	@property
	def _joblib_verbose(self):
		"""joblib Parallel verbosity: 0 unless self.verbose requests joblib output (>=2)."""
		return 1 if self.verbose >= 2 else 0

	def check_coord(self):
		"""
		Checks if target coordinate / name input is valid.

		Returns
		-------
		bool.

		"""

		if ((self.ra is None) or (self.dec is None)) and (self.name is None):
			return False
		else:
			return True

	def _get_gaia(self,maglim=21):
		"""
		Downloads a catalogue of stars in the target region from GAIA DR3.

		Parameters
		----------
		maglim : int,float
			Limiting magnitude in GAIA g band of stars to include in catalogue. The default is 21.
		
		Assigns
		-------
		gaia : DataFrame
			Catalogue of GAIA sources inside cutout region.

		"""
		
		# Get dataframe from Gaia around cutout
		result = Get_Catalogue(self.ra,self.dec,self.flux.shape, Catalog = 'gaia')
		result = result[result.Gmag < maglim]
		result = result.rename(columns={'RA_ICRS': 'ra',
								'DE_ICRS': 'dec',
								'e_RA_ICRS': 'e_ra',
								'e_DE_ICRS': 'e_dec',})
		
		# Convert star RA/DEC to pixel values and input into dataframe
		x,y = self.wcs.all_world2pix(result['ra'].values,result['dec'].values,0)
		result['x'] = x; result['y'] = y

		# Restrict catalogue to only objects inside cutout
		ind = (((x > 0) & (y > 0)) & 
		 	  ((x < (self.flux.shape[2])) & (y < (self.flux.shape[1]))))
		result = result[ind]

		self.gaia = result

	def _assign_phot_method(self,phot_method):
		"""
		Assigns reduction photometry extraction method, currently aperture and PSF photometry is supported.

		Parameters
		----------
		phot_method : str
			Photometry extraction method, either 'aperture' or 'PSF'.

		Raises
		------
		ValueError
			If phot_method is not in the above options.

		Assigns
		-------
		phot_method : str
			Photometry extraction method.

		"""

		if isinstance(phot_method, str):
			method = phot_method.lower()
			if (method == 'psf') | (method == 'aperture'):
				self.phot_method = method
			else:
				m = f'The input method "{method}" is not supported, please select either "psf", or "aperture".'
				raise ValueError(m)
		else:
			m = 'phot_method must be a string equal to either "psf", or "aperture".'
			raise ValueError(m)

	def __clean_lk_cache(self,cache_dir=None):
		if cache_dir is None:
			cache = lk.config.get_cache_dir()


	def get_TESS(self,ra=None,dec=None,name=None,size=None,sector=None,
				 quality_bitmask='default',cache_dir=None,cache=True):
		"""
		Use the lightcurve interface with TESScut to get an FFI cutout 
		of a region around the given coords.

		Parameters
		----------
		ra : float, optional
			RA of the cutout centre. The default is None.
		dec : float, optional
			Dec of the cutout centre. The default is None.
		name : str, optional
			Name of target event in TNF. The default is None.
		size : int, optional
			Size of the cutout (size x size). The default is None.
		sector : int, optional
			Sector to download. The default is None.
		quality_bitmask : str, optional
		 	Parameter for Lightkurve download of TESS TPF. The default is 'default'.
		cache_dir : str, optional
			Directory to cache files. The default is None.

		Raises
		------
		ValueError
			If download process fails for whatever reason.

		Assigns
		-------
		tpf : Lightkurve Target Pixel File
			Tess FFI cutout of the selected region.
		flux : np.array
			Array of flux values stored in tpf.
		wcs : astropy WCS object
			World coordinate system information of the cutout.

		"""
		
		from astropy.coordinates import SkyCoord
		from astropy import units as u

		if sector is None:
			sector = self.sector

		# Find download file from Lightkurve based on coordinate / name
		if (name is None) & (self.name is None):
			if (ra is not None) & (dec is not None):
				c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec) *
								u.degree, frame='icrs')
			else:
				c = SkyCoord(ra=float(self.ra)*u.degree, dec=float(self.dec) *
								u.degree, frame='icrs')
			tess = lk.search_tesscut(c,sector=sector)
		else:
			tess = lk.search_tesscut(name,sector=sector)
		if size is None:
			size = self.size

		# Download 
		tpf = tess.download(quality_bitmask=quality_bitmask,cutout_size=size,download_dir=cache_dir)
		if not cache:
			self._cache_path = tpf.path
		else:
			self._cache_path = None

		# Check to ensure it succeeded
		if tpf is None:
			m = 'Failure in TESScut api, not sure why.'
			raise ValueError(m)
		
		self.tpf = tpf
		self.flux = strip_units(tpf.flux)  # Stripping astropy units so only numbers are returned
		self.flux[np.isnan(self.flux)] = 0
		if self._use_error_image:
			self.eflux = strip_units(tpf.flux_err)
		else:
			self.eflux = None
		self.wcs = tpf.wcs
		self.mjd = tpf.time.mjd
		self.column = tpf.column
		self.row = tpf.row
		self.camera = self.tpf.camera
		self.ccd = self.tpf.ccd


	def make_mask(self,catalogue_path=None,maglim=19,scale=1,strapsize=6,useref=False):
		"""
		Generate a source mask for the cutout region from source catalogues.
		Pixels that are found to include sources will not be included in background calculation.

		Parameters
		----------
		catalogue_path : str, optional
			Local path to source catalogue if using TESSreduce in offline mode. The default is None.
		maglim : float, optional
			Limiting magnitude of sources to include in the source mask. The default is 19.
		scale : float, optional
			Adjusts how much of each source the mask covers. The default is 1.
		strapsize : float, optional
			Width in pixels of the mask for TESS' electrical straps. The default is 6.
		
		Options
		-------
		useref : bool, optional
			Generate the mask solely from the reference frame. The default is False.

		Assigns
		-------
		mask : np.array
			A bitwise source mask for the cutout. Bits are as follows:
				0 - background
				1 - catalogue source
				2 - saturated source
				4 - strap mask
				8 - bad pixel (not used)
				8 - data-driven source (from residual background mask)

		"""

		data = strip_units(self.flux)

		# Generate mask from source catalogue
		if useref:
			mask, cat = Cat_mask(self.ra,self.dec,self.flux.shape,self.wcs,self.flux,self.column,catalogue_path,maglim,scale,strapsize,ref=self.ref,col_offset=self._col_offset)
		else:
			mask, cat = Cat_mask(self.ra,self.dec,self.flux.shape,self.wcs,self.flux,self.column,catalogue_path,maglim,scale,strapsize,ref=self.ref,col_offset=self._col_offset)

		# Generate sky background as the inverse of mask
		sky = ((mask & 1)+1 == 1) * 1.
		sky[sky==0] = np.nan
		tmp = np.nansum(data*sky,axis=(1,2))
		tmp[tmp==0] = 1e12 # random big number 
		ref = data[np.argmin(tmp)] * sky

		## Old code, no longer used 
		## Compute a spatial QE map from a single reference frame.
		## Stored as self.qe_spatial for diagnostic use; the temporal QE
		## correction applied during background() is computed separately by _calc_qe.
		# try:
		# 	self.qe_spatial = correct_straps(ref,mask,parallel=True)
		# except:
		# 	self.qe_spatial = correct_straps(ref,mask,parallel=False)


		c1 = data.shape[1] // 2
		c2 = data.shape[2] // 2
		cmask = np.zeros_like(data[0],dtype=int)
		if self._center_mask:
			cmask[c1-1:c1+2, c2-1:c2+2] = 1

		fullmask = mask | cmask
		sky = ((fullmask & 1)+1 == 1) * 1.
		sky[sky==0] = np.nan
		masked = np.abs(ref*sky)
		mean,med,std = sigma_clipped_stats(masked)# assume sources weight the mean above the bkg
		if useref is False:
			m_second = (masked - mean > 2*std).astype(int)
			self.mask = fullmask | m_second
		else:
			self.mask = fullmask
		self._mask_cat = cat

	def psf_source_mask(self,sigma=5):
		"""
		Generate a source mask by finding PSF like objects in each image.

		Parameters
		----------
		sigma : float, optional
			Photometric spread of source considered for finding sources. The default is 5.

		Assigns
		-------
		prf : TESS PRF Object
			Pixel response function object for this cutout.

		Returns 
		-------
		m : np.array
			Source bitmask.
			
		"""

		from PRF import TESS_PRF

		col = self.column + int(self.size//2) # find column and row, when specifying location on a *say* 90x90 px cutout
		row = self.row + int(self.size//2)

		col += 45 # add on the non-science columns
		row += 1 # add on the non-science row
		if col > 2090:
			col = 2090
		if row > 2040:
			row = 2040

		# Find PRF for cutout (depends on Sector, Camera, CCD, Pixel Row, Pixel Column)
		if self._catalogue_path is not None:

			if self.sector < 4:
				prf = TESS_PRF(self.camera,self.ccd,self.sector,
								col,row,
								localdatadir=f'{self._prf_path}/Sectors1_2_3')
			else:
				prf = TESS_PRF(self.camera,self.ccd,self.sector,
								col,row,
								localdatadir=f'{self._prf_path}/Sectors4+')
		else:
			try:
				prf = TESS_PRF(self.camera,self.ccd,self.sector,
										col,row)
			except Exception as e:
				print(f'Warning: could not load PRF (network error?): {e}')
				return np.ones((self.flux.shape[0], self.flux.shape[1], self.flux.shape[2]))

		self.prf =  prf.locate(5,5,(11,11))

		# Iterate through frames to find PRF like sources
		data = (self._flux_aligned - self.ref) #* mask
		if self.parallel:
			try:
				m = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_source_mask)(frame,self.prf,sigma) for frame in data)
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

	def _calc_qe(self):#,flux_e):
		'''
		Calculate the effective quantum efficiency enhancement of the detector from scattered light.
		'''
		time = deepcopy(self.mjd)
		strap_data = (self.flux) * ((self.mask&4) > 0)*(~self.mask&1)
		with np.errstate(divide='ignore', invalid='ignore'):
			qe = strap_data / self.bkg
		qe[~np.isfinite(qe)] = np.nan
		m,med,std = sigma_clipped_stats(qe,axis=1,sigma_upper=2)
		qes = np.ones_like(qe)
		qes[:,:,:] = med[:,np.newaxis,:]
		qes[np.isnan(qes)] = 1

		av_bkg = np.sum(self.bkg,axis=(1,2))/(self.bkg.shape[1]*self.bkg.shape[2])
		m,med,std = sigma_clipped_stats(av_bkg)
		ind = av_bkg < med + 5*std
		breaks = np.where(np.diff(time[ind]) > 0.5)[0]+1
		breaks = np.insert(breaks, 0, 0)
		breaks = np.append(breaks, len(time[ind]))

		new_qes = deepcopy(qes)
		ind_where = np.where(ind)[0]
		for i in range(len(breaks)-1):
			if abs(breaks[i]-breaks[i+1]) > 100:
				window_size = int(abs(breaks[i]-breaks[i+1])/4)
				if window_size/2 == window_size//2:
					window_size += 1
				seg_idx = ind_where[breaks[i]:breaks[i+1]]
				sav = savgol_filter(qes[ind][breaks[i]:breaks[i+1]], window_size, 1, axis=0)
				new_qes[seg_idx] = sav
		new_qes[new_qes < 1.001] = 1 # set a limit of 1% adjustment
		self.qe = new_qes

	def background(self,gauss_smooth=None,calc_qe=True,strap_iso=True,source_hunt=False,
					interpolate=True,rerun_negative=False,rerun_diff=False,blend_dynamic=False):
		"""
		Calculate the temporal and spatial variation in the background.

		Parameters
		----------
		gauss_smooth : float, optional
			Smoothing factor for the background smoothing. The default is 2.

		Options
		-------
		calc_qe : bool, optional
			Calculate the quantum efficiency in TESS' electrical straps. The default is True.
		strap_iso : bool, optional
			Isolate the electrical straps for calculation. The default is True.
		source_hunt : bool, optional
			Using PSF, search for sources in each frame that may not have been masked out by the catalogue source mask. The default is False.
		interpolate : bool, optional
			Interpolate over masked out objects when calculating background. The default is True.

		Assigns
		-------
		bkg : np.array
			Spatially varying background for each frame.

		"""
		_times = {}
		_t0_total = time.perf_counter()

		if gauss_smooth is None:
			gauss_smooth = self._bkg_gauss_sigma
		# b = np.nansum(self.flux,axis=(1,2))
		# ind = (b < np.percentile(b,24))
		# if calc_qe:
		# 	m,med,s = sigma_clipped_stats((self.flux)[ind] - self.ref,axis=0)
		# else:
		# 	m,med,s = sigma_clipped_stats((self.flux)[ind],axis=0)
		# m,med,std = sigma_clipped_stats(s)

		# m = (s > med + 3*std ) * 1.
		# if calc_qe:
		# 	m = convolve(m,np.ones((5,5)))
		# else:
		# 	m = convolve(m,np.ones((3,3)))

		# m[m>0] = np.nan
		# m = abs(m-1)

		# if strap_iso:
		# 	strap_cols = (self.mask & 4) > 0
		# 	if strap_cols.ndim == 3:
		# 		strap_cols = strap_cols.any(axis=0)
		# 	m[strap_cols] = np.nan

		_t = time.perf_counter()
		if strap_iso:
			m = (self.mask == 0) * 1.
		else:
			m = ((self.mask & 1 == 0) & (self.mask & 2 == 0)) * 1.
		m[m==0] = np.nan

		# Find extra sources not found in catalogue mask
		if source_hunt:
			sm = self.psf_source_mask()
			sm[sm==0] = np.nan
			m = sm * m
		self._bkgmask = m
		_times['mask creation'] = time.perf_counter() - _t

		# Calculate the smooth background 
		if (self.flux.shape[1] > 30) & (self.flux.shape[2] > 30):
			flux = deepcopy(strip_units(self.flux))
			# if calc_qe:
			# 	flux -= self.ref

			bkg_smth = np.zeros_like(flux) * np.nan
			if self.parallel:
				_t = time.perf_counter()
				if self.verbose > 0:
					print('smooth background...')
				bkg_smth = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(Smooth_bkg)(frame,0,interpolate) for frame in flux*m)
				_times['initial smooth background'] = time.perf_counter() - _t
				if self.verbose > 0:
					print('smooth background done')
				if rerun_negative:
					_t = time.perf_counter()
					if self._use_error_image:
						over_sub = (deepcopy(self.flux) - bkg_smth) < -self.eflux # -0.5
					else:
						over_sub = (deepcopy(self.flux) - bkg_smth) <  -0.5
					over_sub = np.nansum(over_sub,axis=0) > 0
					#self._over_sub = over_sub
					#print('overshape ',over_sub.shape)
					#print('m ',m.shape)
					strap_mask = (self.mask & 4) > 0
					if len(strap_mask.shape) == 3:
						strap_mask = strap_mask[0]
					if strap_iso:
						over_sub[strap_mask] = 0
					if source_hunt | (len(self.mask.shape) == 3):
						m[:,over_sub[:,:]] = 1
					else:
						m[over_sub] = 1
					self._bkgmask = m
					if self.verbose > 0:
						print('smooth background rerun (negative correction)...')
					bkg_smth = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(Smooth_bkg)(frame,gauss_smooth,interpolate) for frame in flux*m)
					_times['negative over-subtraction rerun'] = time.perf_counter() - _t
					if self.verbose > 0:
						print('smooth background rerun done')

				if rerun_diff:
					_t = time.perf_counter()
					from photutils.background import Background2D, MedianBackground
					from astropy.stats import SigmaClip
					from scipy.ndimage import label

					bkg_smth = np.array(bkg_smth)
					mean_bkg = np.nanmean(bkg_smth,axis=(1,2))
					low = bkg_smth[mean_bkg < 300]
					fixed_bkg = gaussian_filter(low, sigma=6,axes = 0)
					fixed = flux[mean_bkg < 300] - fixed_bkg
					# np.save('intermediate_flux.npy',fixed)
					# np.save('intermediate_bkg.npy',fixed_bkg)
					m,med,std = sigma_clipped_stats(fixed,axis=0)

					estimator = MedianBackground()
					sc = SigmaClip(sigma=5, maxiters=5)
					try:
						b = Background2D(std,
										box_size=5,
										filter_size=3,
										sigma_clip=sc,
										bkg_estimator=estimator,
										exclude_percentile=50,
										fill_value=0.0)
						std_sub = std - b.background
					except ValueError:
						std_sub = std
					_,smed,sstd = sigma_clipped_stats(std_sub)
					resid_mask = (std_sub > smed + 3*sstd) * 1.0
					ny, nx = resid_mask.shape
					resid_mask = convolve(resid_mask, np.ones((3, 3))) > 1
					labeled, n_comp = label(resid_mask)
					filtered = np.zeros_like(resid_mask)
					for comp in range(1, n_comp + 1):
						pix = labeled == comp
						if pix.sum() > 2000:
							continue
						filtered[pix] = 1
					filtered[:2, :] = 0
					filtered[-2:, :] = 0
					filtered[:, :2] = 0
					filtered[:, -2:] = 0
					resid_mask = filtered
					if source_hunt | (len(self.mask.shape) == 3):
						new_mask = deepcopy(sm)
						new_mask[:,resid_mask[:,:]] = np.nan
					else:
						new_mask = deepcopy(resid_mask) * 1.0
						new_mask[new_mask == 1] = np.nan
						new_mask = abs(new_mask - 1)
					self._bkgmask = new_mask
					bkg_s1 = np.array(bkg_smth)
					if self.verbose > 0:
						print('smooth background rerun (residual surface)...')
					bkg_smth = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(Smooth_bkg)(frame,0,interpolate) for frame in flux*new_mask)
					if blend_dynamic:
						bkg_smth = blend_dynamic_background(bkg_smth, bkg_s1, flux, n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)
					_times['residual surface rerun'] = time.perf_counter() - _t

			else:
				_t = time.perf_counter()
				for i in range(flux.shape[0]):
					bkg_smth[i] = Smooth_bkg((flux*m)[i],0,interpolate)
				_times['initial smooth background'] = time.perf_counter() - _t
		else:
			if self.verbose > 0:
				print('Small tpf, using percentile cut background')
			self.small_background()
			bkg_smth = self.bkg
		
		# Calculate quantum efficiency
		self.bkg = np.array(bkg_smth)
		_t = time.perf_counter()
		if calc_qe:
			self._calc_qe()#,self.eflux)
			self.bkg *= self.qe

		# if calc_qe:
		bkg_pre_fix = np.array(self.bkg)
		from .adaptive_background import get_tessvectors, _interpolate_angles
		_df = get_tessvectors(self.sector, self.camera, data_path=self._vector_path)
		_bkg_median = np.nanmedian(self.bkg, axis=(1, 2))
		if _df is not None:
			_earth_angle, _moon_angle = _interpolate_angles(self.mjd, _df)
			_high_bkg_frames = (_earth_angle < 30.0) | (_moon_angle < 30.0) | (_bkg_median > 300.0)
		else:
			_high_bkg_frames = _bkg_median > 300.0
		if self.verbose > 0:
			print('fixing background anomalies...')
		self.bkg, _sharp_masks, self.bad_bkg = fix_background_anomalies(self.bkg, self.mask,
											flux=deepcopy(self.flux),
											bkg_prev=bkg_pre_fix if blend_dynamic else None,
											bkgmask=self._bkgmask,
											gauss_smooth=gauss_smooth,
											high_bkg_frames=_high_bkg_frames,
											n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)
		_times['anomaly fixing'] = time.perf_counter() - _t
		if self.verbose > 0:
			print('fixing background anomalies done')

		_t = time.perf_counter()
		if self.verbose > 0:
			print('adaptive temporal smoothing...')
		from .adaptive_background import AdaptiveBackground
		smoother = AdaptiveBackground(self.bkg, self.mjd, sector=self.sector, camera=self.camera,
									  data_path=self._vector_path,n_jobs=self.num_cores,backend=self.backend,verbose=self._joblib_verbose)
		if smoother._df is not None:
			smoothed = smoother.smooth(method='savgol').smoothed
			self.bkg = smoothed
		_times['adaptive temporal smoothing'] = time.perf_counter() - _t
		if self.verbose > 0:
			print('adaptive temporal smoothing done')

		# Store data-driven sources from _bkgmask as bit 8, preserving the catalogue mask (bit 1)
		if rerun_diff:
			_t = time.perf_counter()
			#final correction
			f = deepcopy(self.flux)
			f -= self.bkg
			# m, med, std = sigma_clipped_stats(f - self.bkg, axis=(1, 2))
			bkg2d_mask = np.isnan(np.asarray(self._bkgmask))
			if bkg2d_mask.ndim == 3:
				bkg2d_mask = np.any(bkg2d_mask, axis=0)
			bkg_corr = parallel_background2d(f, box_size=9, filter_size=3,
											sigma=3, maxiters=5, n_jobs=self.num_cores,
											mask=bkg2d_mask, backend=self.backend, verbose=self._joblib_verbose)
			self.bkg += bkg_corr

			bkgmask_arr = np.asarray(self._bkgmask)
			if len(self.mask.shape) == 3:
				if bkgmask_arr.ndim == 3:
					new_sources = np.isnan(bkgmask_arr)
				else:
					new_sources = np.isnan(bkgmask_arr)[np.newaxis, :, :]
			else:
				if bkgmask_arr.ndim == 3:
					new_sources = np.any(np.isnan(bkgmask_arr), axis=0)
				else:
					new_sources = np.isnan(bkgmask_arr)
			self.mask = self.mask | (new_sources.astype(self.mask.dtype) * 8)
			_times['final residual correction'] = time.perf_counter() - _t

		if self._timing:
			_total = time.perf_counter() - _t0_total
			print('\nBackground timing report:')
			for _name, _elapsed in _times.items():
				print(f'  {_name:<38s} {_elapsed:6.2f}s  ({100*_elapsed/_total:5.1f}%)')
			print(f'  {"total":<38s} {_total:6.2f}s')

		#self._bkg_temporal_smooth()
		#self._bkg_adaptive_smooth()

	def small_background(self):
		"""
		A different background calculation for if the cutout is too small for high quality results using default process.

		Assigns
		-------
		bkg : np.array
			Spatially varying background for each frame.

		"""
		
		bkg = np.zeros_like(self.flux)
		flux = strip_units(self.flux)
		lim = 2*np.nanmin(flux,axis=(1,2)) #np.nanpercentile(flux,1,axis=(1,2))
		ind = flux > lim[:,np.newaxis,np.newaxis]
		flux[ind] = np.nan
		val = np.nanmedian(flux,axis=(1,2))
		bkg[:,:,:] = val[:,np.newaxis,np.newaxis]
		self.bkg = bkg

	def _bkg_round_3(self,iters=5):
		"""
		Third background calculation.

		Parameters
		----------
		iters : int, optional
			Number of iterations to clean background. The default is 5.

		Assigns
		-------
		bkg : np.array
			Spatially varying background for each frame.

		"""
		
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
				bkg_3 = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(parallel_bkg3)(self.bkg[i],dist_mask[i]) 
															for i in np.arange(len(dist_mask)))
			else:
				bkg_3 = np.zeros_like(self.bkg)
				for i in range(len(dist_mask)):
					bkg_3[i] = parallel_bkg3(self.bkg[i],dist_mask[i])
			self.bkg = np.array(bkg_3)

	def _clip_background(self,sigma=5,ideal_size=90):
		"""
		Performs sigma clip on the background and recomputes clipped points.

		Parameters
		----------
		sigma : float, optional
			Number of sigma to cut background. The default is 5.

		Returns
		-------
		None

		"""
		
		if self.parallel:
			bkg_clip = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(clip_background)(self.bkg[i],self.mask,sigma,ideal_size) 
														for i in np.arange(len(self.bkg)))
		else:
			bkg_clip = np.zeros_like(self.bkg)
			for i in range(len(self.bkg)):
				bkg_clip[i] = clip_background(self.bkg[i],self.mask,ideal_size)
		self.bkg = np.array(bkg_clip)

	def _grad_bkg_clip(self,sigma=3,max_size=1000):
		"""
		Performs sigma clip on the background based on gradients and recomputes clipped points.

		Parameters
		----------
		sigma : float, optional
			Number of sigma to cut background. The default is 3.
		max_size: int, optional
			Maximum allowable size of a region to be clipped.

		Assigns
		-------
		bkg : np.array
			Assigns the recomputed tess background.
		"""
		
		if self.parallel:
			bkg_clip = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(grad_clip_fill_bkg)(self.bkg[i],sigma,max_size) 
														for i in np.arange(len(self.bkg)))
		else:
			bkg_clip = np.zeros_like(self.bkg)
			for i in range(len(self.bkg)):
				bkg_clip[i] = grad_clip_fill_bkg(self.bkg[i],max_size)
		self.bkg = np.array(bkg_clip)

	def _bkg_temporal_smooth(self,window_size=None):
		if window_size is None:
			window_size = self._bkg_temporal_window # still need to decide what this is 
			
		#grad = np.gradient(np.sum(self.bkg,axis=(1,2)),self.mjd) # not sure about this gradient 
		av_bkg = np.nanmean(self.bkg,axis=(1,2))
		time = deepcopy(self.mjd)
		m,med,std = sigma_clipped_stats(av_bkg)
		ind = av_bkg < med + 5*std
		# m,med,std = sigma_clipped_stats(abs(grad))
		# ind = abs(grad) < (med + 5*std)
		# left  = np.concatenate([[False], ind[:-1]])
		# right = np.concatenate([ind[1:],  [False]])
		# ind   = ind & (left | right)
		# from scipy.ndimage import label
		# clusters, _ = label(ind)
		# sizes = np.bincount(clusters.ravel())   # sizes[i] = number of points in cluster i
		# small = sizes < window_size
		# ind[small[clusters]] = False 
		# if np.sum(ind) == 0:
		# 	window_size = int(0.5 * np.max(sizes))
		# 	if window_size % 2 == 0:
		# 		window_size += 1
		# 	print(f'!!!WARNING window size for temporal background smoothing decreaesd to {window_size}')
		# 	ind = abs(grad) < (med + 5*std)
		# 	small = sizes < window_size
		# 	ind[small[clusters]] = False 

		ind_where = np.where(ind)[0]

		breaks = np.where(np.diff(time[ind]) > 0.5)[0]+1
		breaks = np.insert(breaks, 0, 0)
		breaks = np.append(breaks, len(time[ind]))
		new_bkg = deepcopy(self.bkg)  # shape (T, X, Y)

		for i in range(len(breaks) - 1):
			if abs(breaks[i]-breaks[i+1]) > 100:
				window_size = int(abs(breaks[i]-breaks[i+1])/4)
				if window_size/2 == window_size//2:
					window_size += 1
				seg_idx = ind_where[breaks[i]:breaks[i+1]]
				seg = new_bkg[seg_idx]									  # (n_seg, X, Y)
				sav = savgol_filter(seg, window_size, 1, axis=0)					 # (n_seg, X, Y)

				# per-pixel residual threshold
				resid = np.abs(seg - sav)							   # (n_seg, X, Y)
				threshold = np.median(resid, axis=0) + 5 * np.std(resid, axis=0)  # (X, Y)

				exceeds = resid > threshold[np.newaxis]					 # (n_seg, X, Y)

				# leading bad run: cumprod stays 1 only while all preceding values were True
				start_clip = np.cumprod(exceeds,axis=0).sum(axis=0)  # (X, Y)
				end_clip = len(seg_idx) - np.cumprod(exceeds[::-1], axis=0).sum(axis=0)

				# use sav only within [start_clip, end_clip), raw outside
				t = np.arange(len(seg_idx))[:, np.newaxis, np.newaxis]	  # (n_seg, 1, 1)
				use_sav = (t >= start_clip) & (t < end_clip)

				new_bkg[seg_idx] = np.where(use_sav, sav, seg)

		flux = deepcopy(self.flux) - new_bkg
		# med = sigma_clipped_stats(flux,axis=(1,2))[1]
		med = np.median(flux.reshape(len(flux), -1), axis=1)
		new_bkg += med[:,np.newaxis,np.newaxis]
		self.bkg = new_bkg


	def _bkg_adaptive_smooth(self, gap_thresh=0.5, clip_sigma=5.0,
							 plot=None, savename=None):
		"""
		Adaptive temporal Gaussian smoothing of the background cube.

		Two Gaussian filters are computed per segment — one wide (sigma=n/4,
		for stable epochs) and one narrow (sigma=n/32, for rapidly varying
		epochs) — and linearly blended per-frame based on the local background
		gradient estimated from a short first-pass SavGol smooth.

		  alpha = 1  →  wide filter   (stable background, low gradient)
		  alpha = 0  →  narrow filter (variable background, high gradient)

		On simple/stable data (e.g. GRB fields) the gradient is near-zero
		everywhere so alpha ≈ 1 and the result matches SavGol n//4.
		On complex scattered-light data the gradient is elevated during ramps
		so alpha → 0 and the narrower filter tracks the variation.

		Parameters
		----------
		gap_thresh : float
			Time gap in days that marks a segment boundary. Default 0.5.
		clip_sigma : float
			Sigma threshold for outlier frame rejection. Default 5.0.
		"""
		import matplotlib.pyplot as plt
		if plot is None:
			plot = self.diagnostic_plot
		if savename is None:
			savename = self.savename

		av_bkg = np.nanmean(self.bkg, axis=(1, 2))
		av_bkg_raw = av_bkg.copy()
		time = deepcopy(self.mjd)

		_, med, std = sigma_clipped_stats(av_bkg)
		ind = av_bkg < med + clip_sigma * std
		ind_where = np.where(ind)[0]

		breaks = np.where(np.diff(time[ind]) > gap_thresh)[0] + 1
		breaks = np.insert(breaks, 0, 0)
		breaks = np.append(breaks, len(time[ind]))

		new_bkg = deepcopy(self.bkg)
		alpha_all = np.full(len(time), np.nan)
		grad_all = np.full(len(time), np.nan)

		for i in range(len(breaks) - 1):
			seg_idx = ind_where[breaks[i]:breaks[i + 1]]
			n = len(seg_idx)
			if n < 10:
				continue

			seg = new_bkg[seg_idx]	# (n, X, Y)
			av  = av_bkg[seg_idx]	 # (n,)

			# SavGol window sizes — wide for stable regions, narrow for variable.
			# SavGol is strictly local (±window/2 frames) so it does not
			# average in distant frames the way a Gaussian filter does.
			w_wide = max(n // 4, 5)
			if w_wide % 2 == 0:
				w_wide += 1
			w_narrow = max(n // 8, 5)
			if w_narrow % 2 == 0:
				w_narrow += 1

			# ── Pass 1: gradient from short SavGol of spatial mean ───────
			rw = max(min(n // 32, 51), 5)
			if rw % 2 == 0:
				rw += 1
			av_rough = savgol_filter(av, rw, 1)
			grad = np.abs(np.gradient(av_rough))
			gw = max(min(n // 32, 25), 3)
			grad_smooth = np.convolve(grad, np.ones(gw) / gw, mode='same')
			med_grad = float(np.median(grad_smooth))

			# ── Blend weight: alpha=1 (wide) where gradient is low ───────
			if med_grad < 1e-10:
				alpha = np.ones(n)
			else:
				rate_norm = grad_smooth / med_grad
				alpha = np.clip(1.0 / np.maximum(rate_norm, 1.0), 0.0, 1.0)

			alpha_all[seg_idx] = alpha
			grad_all[seg_idx] = grad_smooth

			# ── Pass 2: blend wide and narrow SavGol filters ─────────────
			sav_wide = savgol_filter(seg, w_wide, 1, axis=0)
			sav_narrow = savgol_filter(seg, w_narrow, 1, axis=0)
			a = alpha[:, np.newaxis, np.newaxis]
			new_bkg[seg_idx] = a * sav_wide + (1.0 - a) * sav_narrow

		# ── Median flux correction (background pixels only) ──────────────
		# Use only source-free pixels so that star/galaxy flux does not
		# bias the correction upward.
		bkg_pixel_mask = ~(self.mask & 1).astype(bool)   # True = background
		flux = deepcopy(self.flux) - new_bkg		 # (T, X, Y)
		flux_bkg = flux.copy().astype(float)
		flux_bkg[:, ~bkg_pixel_mask] = np.nan
		med = np.nanmedian(flux_bkg, axis=(1, 2))
		new_bkg += med[:, np.newaxis, np.newaxis]
		self.bkg = new_bkg

		if plot:
			av_smooth = np.nanmean(self.bkg, axis=(1, 2))
			t = self.mjd.copy()

			gap_idx = np.where(np.diff(t) > gap_thresh)[0]
			def _nan_gaps(arr):
				a = arr.copy().astype(float)
				if len(gap_idx):
					a[gap_idx] = np.nan
				return a

			fig, axes = plt.subplots(3, 1,
									 figsize=(1.5 * fig_width, 3.0 * fig_width),
									 sharex=True)

			# Panel 1: background + wide / narrow / adaptive smooths
			ax = axes[0]
			ax.plot(t, av_bkg_raw, '.k', ms=1.5, alpha=0.3, label='Raw')
			ax.plot(t, _nan_gaps(av_smooth), lw=1.5, label='Adaptive blend')
			ax.set_ylabel('Mean bkg (e⁻/s)', fontsize=11)
			ax.legend(fontsize=9)
			ax.set_title('Background adaptive Gaussian blend', fontsize=11)

			# Panel 2: gradient and blend weight
			ax = axes[1]
			col_g = 'C1'; col_a = 'C2'
			ax.plot(t, _nan_gaps(grad_all), color=col_g, lw=1,
					label='|d(rough bkg)/dt|')
			ax.set_ylabel('Gradient (e⁻/s/frame)', fontsize=11, color=col_g)
			ax.tick_params(axis='y', labelcolor=col_g)
			axr = ax.twinx()
			axr.plot(t, _nan_gaps(alpha_all), color=col_a, lw=1, alpha=0.8,
					 label='α (blend weight)')
			axr.set_ylabel('α  (1=wide, 0=narrow)', fontsize=11, color=col_a)
			axr.tick_params(axis='y', labelcolor=col_a)
			axr.set_ylim(-0.05, 1.05)
			ax.set_title('High gradient → α→0 (narrow filter)', fontsize=11)

			# Panel 3: residuals
			ax = axes[2]
			ax.plot(t, _nan_gaps(av_bkg_raw - av_smooth), '.k', ms=1.5, alpha=0.4,
					label='Residual (raw − smooth)')
			ax.axhline(0, color='r', lw=0.8, ls='--')
			ax.set_ylabel('Residual (e⁻/s)', fontsize=11)
			ax.set_xlabel('Time (MJD)', fontsize=11)
			ax.legend(fontsize=9)

			plt.tight_layout()
			plt.show()
			if savename is not None:
				plt.savefig(savename + '_bkg_smooth.pdf', bbox_inches='tight')

	def get_ref(self,start = None, stop = None):
		"""
		Get reference image to use for subtraction and mask creation.
		The image is made from all images with low background light.

		Parameters
		----------
		start : int, optional
			First frame to consider for reference determination. The default is None.
		stop : int, optional
			Final frame to consider for reference determination. The default is None.

		Assigns
		-------
		ref : np.array
			Reference image for the cutout.
		ref_ind : int
			Index pointing to which frame was used for reference.

		"""

		data = strip_units(self.flux)

		if self._force_ref_ind is not None:
			self.ref_ind = self._force_ref_ind
			self.ref = deepcopy(data[self._force_ref_ind])
		else:
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
			summed = np.nanmedian(d,axis=(1,2))
			summed[summed <=0] = 1e5
			lim = np.argmin(summed)
			lim = np.percentile(summed[np.isfinite(summed)],5)
			summed[summed>lim] = 0
			inds = np.where(ind)[0]
			ref_ind = start + inds[np.argmax(summed)]
			reference = deepcopy(data[ref_ind])
			if len(reference.shape) > 2:
				reference = reference[0]
				ref_ind = ref_ind[0]
			reference[reference <= 0] = np.nan
			base = np.nanmin(reference)
			# reference -= base
			self.ref = reference 
			self.ref_ind = ref_ind
			# self.flux -= base

	def stack_ref(self,time_restriction=None):
		if time_restriction is None:
			time_restriction = self._ref_time_window
		# m,med,std = sigma_clipped_stats(self.flux,axis=(1,2))
		_p = self.flux.reshape(len(self.flux), -1)
		med = np.median(_p, axis=1)
		std = np.median(np.abs(_p - med[:, np.newaxis]), axis=1) * 1.4826
		# sm, smed, sstd = sigma_clipped_stats(std)
		smed = np.median(std)
		sstd = np.median(np.abs(std - smed)) * 1.4826
		ind = np.where((std < (smed + 3*sstd)) & (std > (smed - 3*sstd)))[0]
		times = self.mjd[ind]
		ref_time = self.mjd[self.ref_ind]
		good = abs(times - ref_time) <= time_restriction
		ind = ind[good]
		stack = np.nanmedian(self.flux[ind],axis=0)
		self.ref = stack


	def centroids_shifts_starfind(self,plot=None,savename=None):
		"""
		Depricated.
		Calculate the centroid shifts of sources for time series images using Starfinding based on TESS PRF.

		Options
		----------
		plot : bool, optional
			Plot a diagnostic figure for the shift calculation. The default is None.
		savename : str, optional
			Save name for output. The default is None.

		Assigns
		-------
		shift : np.array
			Median x,y shift of sources over the time series.

		"""

		from PRF import TESS_PRF
		from photutils.detection import StarFinder

		if plot is None:
			plot = self.diagnostic_plot
		if savename is None:
			savename = self.savename

		# hack solution for new lightkurve
		f = strip_units(self.flux)
		m = self.ref.copy()

		mean, med, std = sigma_clipped_stats(m, sigma=3.0)

		prf = TESS_PRF(self.camera,self.ccd,self.sector,
							self.column+self.flux.shape[2]/2,self.row+self.flux.shape[1]/2)
		self.prf =  prf.locate(5,5,(11,11))
		
		finder = StarFinder(2*std,kernel=self.prf,exclude_border=True)
		s = finder.find_stars(m-med)
		
		mx = s['xcentroid']
		my = s['ycentroid']
		x_mid = self.flux.shape[2] / 2
		y_mid = self.flux.shape[1] / 2
		
		self._dat_sources = s.to_pandas()
		
		if self.parallel:
			shifts = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(
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
		self.shift = meds #smooth
		
		if plot:
			t = self.mjd
			ind = np.where(np.diff(t) > .5)[0]
			smooth[ind,:] = np.nan
			plt.figure(figsize=(1.5*fig_width,1*fig_width))
			plt.plot(t,meds[:,1],'.',label='Row shift',alpha =0.5)
			plt.plot(t,smooth[:,1],'-',label='Smoothed row shift')
			plt.plot(t,meds[:,0],'.',label='Col shift',alpha =0.5)
			plt.plot(t,smooth[:,0],'-',label='Smoothed col shift')
			plt.ylabel('Shift (pixels)',fontsize=15)
			plt.xlabel('Time (MJD)',fontsize=15)
			plt.legend()
			plt.show()
			if savename is not None:
				plt.savefig(savename+'_disp.pdf', bbox_inches = "tight")
		
	def fit_shift(self,smooth=True,plot=None,savename=None):
		"""
		Calculate the centroid shifts of sources for time series images 
		by finding the shifts which minimize the difference between frames and reference.

		Options
		----------
		plot : bool, optional
			Plot a diagnostic figure for the shift calculation. The default is None.
		savename : str, optional
			Save name for output. The default is None.

		Assigns
		-------
		shift : np.array
			x,y shift of sources over the time series.

		"""
		import matplotlib.pyplot as plt

		if plot is None:
			plot = self.diagnostic_plot
		if savename is None:
			savename = self.savename
		
		# sources = ((self.mask & 1) ==1) * 1.0 - (convolve((self.mask & 2),np.ones((3,3))) > 0) * 1.0
		sources = ((self.mask & 1) ==1) * 1.0 - (self.mask & 2) * 1.0
		sources[sources<=0] = 0
		sources[self.mask.shape[0]-3:self.mask.shape[0]+4,self.mask.shape[1]-3:self.mask.shape[1]+4] = 0

		f = deepcopy(self.flux)
		m = self.ref.copy() * sources
		m[m==0] = np.nan
		# f[f > 1e4] = np.nan
		#eref = self.eflux[self.ref_ind]


		if self.parallel:
			ind = np.arange(len(f))
			shifts = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(
						#delayed(difference_shifts)(f[i],m,self.eflux[i],eref) for i in ind)
						delayed(difference_shifts)(f[i],m) for i in ind)
			shifts = np.array(shifts)
		else:
			shifts = np.zeros((len(f),2)) * np.nan
			for i in range(len(f)):
				#shifts[i,:] = difference_shifts(f[i],m,self.eflux[i],eref)
				shifts[i,:] = difference_shifts(f[i],m)
		sraw = deepcopy(shifts)
		if smooth:
			shifts = Smooth_motion(shifts,self.tpf)

		if self.shift is not None:
			self.shift += shifts
		else:
			self.shift = shifts

		if plot:
			t = self.mjd
			ind = np.where(np.diff(t) > .5)[0]
			shifts[ind,:] = np.nan
			plt.figure(figsize=(1.5*fig_width,1*fig_width))
			plt.plot(t,sraw[:,0],'.',label='Row shift',alpha = 0.5)
			plt.plot(t,sraw[:,1],'.',label='Col shift',alpha = 0.5)
			if smooth:
				plt.plot(t,shifts[:,0],'-',label='Smoothed row shift')
				plt.plot(t,shifts[:,1],'-',label='Smoothed col shift')
			plt.ylabel('Shift (pixels)',fontsize=15)
			plt.xlabel('Time (MJD)',fontsize=15)
			plt.legend()
			plt.show()
			if savename is not None:
				plt.savefig(savename+'_disp_corr.pdf', bbox_inches = "tight")

	def plot_shifts(self,savename=None):
		import matplotlib.pyplot as plt
		t = self.mjd
		shifts = self.shift
		ind = np.where(np.diff(t) > .5)[0]
		shifts[ind,:] = np.nan
		plt.figure(figsize=(1.5*fig_width,1*fig_width))
		plt.plot(t,shifts[:,0],'.',label='Row shift',alpha =0.5)
		plt.plot(t,shifts[:,1],'.',label='Col shift',alpha =0.5)
		plt.ylabel('Shift (pixels)',fontsize=15)
		plt.xlabel('Time (MJD)',fontsize=15)
		plt.legend()
		plt.show()
		plt.tight_layout()
		if savename is not None:
			plt.savefig(savename+'_alignment.pdf', bbox_inches = "tight")

	def shift_images(self,median=False):
		"""
		Shifts each target image to the reference using the values given in offset. Breaks horribly if data is all 0.

		Options
		----------
		median : bool, optional
			Shift the reference to the target images using the reverse of the shifts. The default is False.

		Assigns
		-------
		flux : np.array
			Array of flux values now shifted to be best alignment.

		"""

		from .helpers import _shift_one, _shift_ref_one
		shifted = self.flux.copy()
		nans = ~np.isfinite(shifted)
		shifted[nans] = 0.
		if median:
			if self.parallel:
				result = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(
					delayed(_shift_ref_one)(self.ref, shifted[i], self.shift[i])
					for i in range(len(shifted)))
				shifted = np.array(result)
			else:
				for i in range(len(shifted)):
					shifted[i] = _shift_ref_one(self.ref, shifted[i], self.shift[i])
			self.flux -= shifted

		else:
			if self.parallel:
				result = Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(
					delayed(_shift_one)(shifted[i], self.shift[i])
					for i in range(len(shifted)))
				self.flux = np.array(result)
			else:
				for i in range(len(shifted)):
					shifted[i] = _shift_one(shifted[i], self.shift[i])
				self.flux = shifted
		
	def bin_data(self,lc=None,time_bin=6/24,frames = None):
		"""
		Bin a light curve to the desired duration specified by bin_size

		Parameters
		----------
		lc : TYPE, optional
			DESCRIPTION. The default is None.
		time_bin : TYPE, optional
			DESCRIPTION. The default is 6/24.
		frames : TYPE, optional
			DESCRIPTION. The default is None.

		Returns
		-------
		binlc : TYPE
			DESCRIPTION.

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
		Bin a light curve flux to the desired duration specified by bin_size

		Parameters
		----------
		flux : TYPE, optional
			DESCRIPTION. The default is None.
		time_bin : TYPE, optional
			DESCRIPTION. The default is 6/24.
		frames : TYPE, optional
			DESCRIPTION. The default is None.

		Returns
		-------
		binf : TYPE
			DESCRIPTION.
		bint : TYPE
			DESCRIPTION.

		"""

		if flux is None:
			flux = self.flux

		t = self.mjd

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

	def check_trend(self,lc=None,limit=0.6):
		from scipy.stats import pearsonr
		if lc is None:
			lc = self.lc[1]
		#print('!!!! ',lc.shape)
		finite = np.isfinite(lc) & np.isfinite(self.shift[:,0]) & np.isfinite(self.shift[:,1])
		p1 = pearsonr(lc[finite],self.shift[finite,0])[0]
		p2 = pearsonr(lc[finite],self.shift[finite,1])[0]
		if (abs(p1) > limit) | (abs(p2) > limit):
			if abs(p1) > abs(p2):
				m = 'x shift'
				p = p1
			else:
				m = 'y shift'
				p = p2
			print(f'High correlation between lc and {m}: {np.round(p,2)}')
		return np.max([p1,p2])


	def diff_lc(self,time=None,x=None,y=None,ra=None,dec=None,tar_ap=3,
				sky_in=5,sky_out=9,phot_method=None,psf_snap='brightest',
				bkg_poly_order=3,plot=None,savename=None,mask=None,diff = True):
		"""
		Calculate the difference imaged light curve. if no position is given (x,y or ra,dec)
		then it degaults to the centre. Sky flux is calculated with an annulus aperture surrounding 
		the target aperture and subtracted from the source. The sky aperture undergoes sigma clipping
		to remove pixels that are poorly subtracted and contain other sources.

		Parameters
		----------
		time : TYPE, optional
			DESCRIPTION. The default is None.
		x : TYPE, optional
			DESCRIPTION. The default is None.
		y : TYPE, optional
			DESCRIPTION. The default is None.
		ra : TYPE, optional
			DESCRIPTION. The default is None.
		dec : TYPE, optional
			DESCRIPTION. The default is None.
		tar_ap : TYPE, optional
			DESCRIPTION. The default is 3.
		sky_in : TYPE, optional
			DESCRIPTION. The default is 5.
		sky_out : TYPE, optional
			DESCRIPTION. The default is 9.
		phot_method : TYPE, optional
			DESCRIPTION. The default is None.
		plot : TYPE, optional
			DESCRIPTION. The default is None.
		savename : TYPE, optional
			DESCRIPTION. The default is None.
		mask : TYPE, optional
			DESCRIPTION. The default is None.
		diff : TYPE, optional
			DESCRIPTION. The default is True.

		Returns
		-------
		lc : TYPE
			DESCRIPTION.
		sky : TYPE
			DESCRIPTION.

		"""
		import matplotlib.pyplot as plt

		if plot is None:
			plot = self.diagnostic_plot
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
			
		if (ra is not None) & (dec is not None):
			x,y = self.wcs.all_world2pix(ra,dec,0)
			x = int(np.round(np.ravel(x)[0],0))
			y = int(np.round(np.ravel(y)[0],0))
		elif (x is None) & (y is None):
			x,y = self.wcs.all_world2pix(self.ra,self.dec,0)
			x = int(np.round(np.ravel(x)[0],0))
			y = int(np.round(np.ravel(y)[0],0))

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
			if self._use_error_image:
				tar_err = np.nansum((self.eflux)*ap_tar,axis=(1,2))#sky_std * tar_ap**2
			else:
				tar_err = sky_std * tar_ap**2
		if phot_method == 'psf':
			if psf_snap is None:
				psf_snap = 'brightest'

			tar, tar_err = self.psf_photometry(x,y,diff=diff,snap=psf_snap,bkg_poly_order=bkg_poly_order)
		elif phot_method == 'photutils':
			if psf_snap is None:
				psf_snap = 'brightest'
			tar, tar_err = self.psf_photutils(xPix=x, yPix=y, snap=psf_snap)
		nan_ind = np.where(np.nansum(self.flux,axis=(1,2))==0,True,False)
		nan_ind[self.ref_ind] = False
		tar[nan_ind] = np.nan
		tar_err[nan_ind] = np.nan

		# ── Orbit ref flux correction ──────────────────────────────────────────
		if self.orbit_ref and hasattr(self, 'orbit_refs') and hasattr(self, 'orbit_segments'):
			orb_flux = {}
			orb_err = {}
			for seg, ref_im in self.orbit_refs.items():
				f = np.nansum(ref_im * ap_tar) - np.nanmedian(ref_im * ap_sky) * tar_ap**2
				sky_vals = ref_im * ap_sky
				e = np.nanstd(sky_vals[np.isfinite(sky_vals)]) * tar_ap**2
				orb_flux[seg] = f
				orb_err[seg] = e

			# primary = orbit whose ref has lowest stddev
			primary_seg = min(self.orbit_refs, key=lambda s: np.nanstd(self.orbit_refs[s]))
			f_primary = orb_flux[primary_seg]
			e_primary = orb_err[primary_seg]

			for seg in self.orbit_refs:
				if seg == primary_seg:
					continue
				delta = orb_flux[seg] - f_primary
				delta_err = np.sqrt(orb_err[seg]**2 + e_primary**2)
				if np.abs(delta) > delta_err:
					mask = self.orbit_segments == seg
					tar[mask] += delta

		time = self.mjd

		lc = np.array([time, tar, tar_err])
		sky = np.array([time, sky_med, sky_std])
		
		if plot:
			self.dif_diag_plot(ap_tar,ap_sky,lc = lc,sky=sky,data=data)
			plt.show()
			if savename is not None:
				plt.savefig(savename + '_diff_diag.pdf', bbox_inches = "tight")

		#p = self.check_trend(lc=lc[1])
		return lc, sky

	def dif_diag_plot(self,ap_tar,ap_sky,lc=None,sky=None,data=None):
		"""
		Makes a plot showing the target light curve, sky, and difference image at the brightest point
		in the target lc.

		Parameters
		----------
		ap_tar : array
			Aperture to perform photometry on.
		ap_sky : array
			Aperture to perform sky photometry on.
		lc : array, optional
			Light curve of the target to plot. If None, then the assigned lc is used. The default is None.
		sky : array, optional
			Light curve of the sky. The default is None.
		data : array, optional
			Array of images to use in creating the light curve. The default is None.

		Returns
		-------
		Figure.

		"""
		import matplotlib.pyplot as plt

		if lc is None:
			lc = self.lc
		if sky is None:
			sky = self.sky
		if data is None:
			data = self.flux
		plt.figure(figsize=(3*fig_width,1*fig_width))
		plt.subplot(121)
		plt.fill_between(lc[0],sky[1]-sky[2],sky[1]+sky[2],alpha=.5,color='C1')
		plt.plot(sky[0],sky[1],'C3.',label='Sky')
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
		except (IndexError, ValueError):
			return
		d = data[maxind]
		nonan1 = np.isfinite(d)
		nonan2 = np.isfinite(d*ap)
		plt.imshow(data[maxind],origin='lower',
					vmin=np.nanpercentile(d,8),
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

		Parameters
		----------
		lc : array, optional
			light curve to plot. The default is None.
		ax : matplotlib axes, optional
			axes to plot onto. The default is None.
		ground : Bool, optional
			If True, then ZTF data will be plotted alongside TESS. The default is False.
		time_bin : float, optional
			Length of time in days to bin data . The default is 6/24.
		xlims : list, optional
			List of x limits to add to the plot. The default is None.

		Returns
		-------
		Figure.

		"""
		import matplotlib.pyplot as plt

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

		Parameters
		----------
		filename : str
			Name for the saved file.
		time_bin : float, optional
			Timeframe in days to bin the TESS data. The default is None.

		Returns
		-------
		csv file.

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

		Parameters
		----------
		lc : array, optional
			Light curve to convert. The default is None.
		flux_unit : str, optional
			Flux units of the light curve. The default is None.
			Valid options:
				counts
				mjy
				cgs

		Returns
		-------
		light : lightkurve object
			The input lightcurve wrapped in the lightkurve format.

		"""
		from astropy import units as u
		from astropy.time import Time

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
								 corr_correction,imaging):
		"""
		Updates relevant parameters for if reduction functions are called out of order.

		Parameters
		----------
		align : Bool
			Trigger alignment procedure.
		parallel : Bool
			Run in parallel.
		calibrate : Bool
			Calibrate the data.
		plot : Bool
			plot the lightcurve.
		diff_lc : Bool
			Create the differenced light curve .
		diff : Bool
			Run difference imaging.
		verbose : int
			Set verbosity.
		corr_correction : Bool
			Run the background correlation correction step .

		Assigns
		-------
		input options

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
		if imaging is not None:
			self.imaging = imaging


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

		Parameters
		----------
		limit : float, optional
			Corrects for correlation coefficents larger than limit. 
			If the flux and the background of tess are correlated (absolute value of correlation coefficent, |r|) 
			to a level higher than limit, a fit to minimize this coefficent is preformed, 
			and the new background and flux values are returned. Default is 0.8.

		Returns
		-------
		None.

		"""
		flux, bkg = multi_correlation_cor(self,limit=limit,cores=self.num_cores)
		self.flux = flux 
		self.bkg = bkg

	def _psf_initialise(self,cutoutSize,loc,time_ind=None,ref=False):
		"""
		For gathering the cutouts and PRF base.

		Parameters
		----------
		cutoutSize : int
			Size of the cutouts in pixels.
		loc : array_like
			Pixel coordinates to evaluate.
		ref : bool, optional
			Toggles whether the reference image is used for calculations. The default is False.

		Returns
		-------
		prf : TESS_PRF Class object
			The effective point-spread function generated from TESS_PRF.
		cutout : TYPE
			DESCRIPTION.

		"""
		if time_ind is None:
			time_ind = np.arange(0,len(self.flux))

		
		col = self.column - int(self.size//2) + loc[0] # find column and row, when specifying location on a *say* 90x90 px cutout
		row = self.row - int(self.size//2) + loc[1] 

		if isinstance(loc[0], (float, np.floating, np.float32, np.float64)):
			loc[0] = int(np.round(loc[0],0))
		if isinstance(loc[1], (float, np.floating, np.float32, np.float64)):
			loc[1] = int(np.round(loc[1]))

		col += 45 # add on the non-science columns
		row += 1 # add on the non-science row
		if col > 2090:
			col = 2090
		if row > 2040:
			row = 2040
		row = np.max([row,10])
		col = np.max([col,45])	
		try:	
			prf = TESS_PRF(self.camera,self.ccd,self.sector,col,row) # initialise psf kernel
		except:
			print(self.camera,self.ccd,self.sector,col,row)
			raise ValueError
		if ref:
			cutout = (self.flux+self.ref)[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
		else:
			cutout = self.flux[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
		if self._use_error_image:
			ecutout = self.eflux[time_ind,loc[1]-cutoutSize//2:loc[1]+1+cutoutSize//2,loc[0]-cutoutSize//2:loc[0]+1+cutoutSize//2] # gather cutouts
		else:
			ecutout = np.ones_like(cutout) * 0.1
		#else:
		return prf, cutout, ecutout

	def moving_psf_photometry(self,xpos,ypos,size=5,time_ind=None,xlim=2,ylim=2):
		"""
		PSF photometry for moving targets.

		Parameters
		----------
		xpos : array_like
			x pixel locations for the initial guess of the target region.
		ypos : array_like
			y pixel locations for the initial guess of the target region.
		size : int, optional
			Size of pixel cutout to use (should be odd). The default is 5.
		time_ind : array_like, optional
			Indices of the time series to use. The default is None.
		xlim : int, optional
			Width of the cutout in pixels. The default is 2.
		ylim : int, optional
			Height of the cutout in pixels. The default is 2.
		
		Returns
		-------
		flux : array_like
			Flux light curve across entire sector.
		"""
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
			prfs, cutouts, ecutouts = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_initialise)(self.flux,self.camera,self.ccd,
																						 self.sector,self.column,self.row,
																					 size,[xpos[i],ypos[i]],time_ind) for i in inds))
		else:
			prfs = []
			cutouts = []
			for i in range(len(time_ind)):
				prf, cutout, ecutouts = self._psf_initialise(size,[xpos[i],ypos[i]],time_ind=time_ind[i])
				prfs += [prf]
				cutouts += [cutout]
		cutouts = np.array(cutouts)
		print('made cutouts')
		if self.parallel:
			flux, pos = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_full)(cutouts[i],prfs[i],self.shift[i],xlim,ylim) for i in inds))
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

	def psf_photutils(self,xPix=None,yPix=None,size=5,local_bkg=False,epsf=None,
					  flux=None,eflux=None,ref=False,return_pos=False,
					  snap='brightest'):
		from photutils.psf import PSFPhotometry
		from astropy.table import Table
		rad = size // 2

		if flux is None:
			flux = self.flux
		if (flux.shape[1] < size) | (flux.shape[2] < size):
			e = 'Image dimensions must be larger than the cutout size'
			raise ValueError(e)
		if eflux is None:
			eflux = self.eflux
		if eflux is None:
			f = strip_units(flux)
			eflux = np.nanstd(f, axis=(1, 2))[:, np.newaxis, np.newaxis] * np.ones_like(f)

		if (xPix is None) | (yPix is None):
			xPix = flux.shape[2]//2
			yPix = flux.shape[1]//2

		if epsf is None:
			if self.epsf is None:
				col = self.column - int(self.size//2) + yPix # find column and row, when specifying location on a *say* 90x90 px cutout
				row = self.row - int(self.size//2) + xPix
				col += 45 # add on the non-science columns
				row += 1 # add on the non-science row
				if col > 2090:
					col = 2090
				if row > 2040:
					row = 2040

				self.epsf = simulate_epsf(self.camera,self.ccd,self.sector,col,row)
			epsf = self.epsf

		if local_bkg:
			localbkg_estimator = LocalBackground(1.5, 7, bkgstat)
		else:
			localbkg_estimator = None
		
		if ref:
			flux = flux + self.ref

		cutouts = flux[:,yPix-rad:yPix+rad+1,xPix-rad:xPix+rad+1]
		ecutouts = eflux[:,yPix-rad:yPix+rad+1,xPix-rad:xPix+rad+1]
		fit_shape = (size, size)
		psfphot = PSFPhotometry(epsf, fit_shape, finder=None,aperture_radius=1.5,
								localbkg_estimator=localbkg_estimator,xy_bounds=(0.05))
		init = Table()
		init['x_init'] = [size//2]
		init['y_init'] = [size//2]
		if snap.lower() != 'all':
			if snap.lower() == 'brightest':
				weight = np.abs(np.nansum((cutouts/ecutouts)[:,1:4,1:4],axis=(1,2)))
				weight[np.isnan(weight)] = 0
				ind = np.argmax(weight)
				rcut = cutouts[ind] 
				ecut = ecutouts[ind]
			elif snap.lower() == 'ref':
				rcut = self.ref[:,yPix-rad:yPix+rad+1,xPix-rad:xPix+rad+1]
				ecut = ecutouts[self.ref_ind]

			
			phot = psfphot(rcut, error=ecut,init_params=init)
			# fit with the best position
			init2 = Table()
			init['x_init'] = phot['x_fit']
			init['y_init'] = phot['y_fit']	
			flux = np.zeros(len(self.flux)) * np.nan
			eflux = np.zeros(len(self.flux)) * np.nan
			psfphot2 = PSFPhotometry(epsf, fit_shape, finder=None,aperture_radius=1.5,
									 xy_bounds=(0.05),localbkg_estimator=localbkg_estimator)
			f,ef = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(parallel_photutils)(cutouts[i],ecutouts[i],psfphot2,init) for i in np.arange(len(cutouts))))
			f = np.array(f).flatten()
			ef = np.array(ef).flatten()
			phot = phot.to_pandas()
			pos = phot[['x_fit','y_fit']].values + np.array([xPix,yPix]) - size//2
			epos = phot[['x_err','y_err']].values
		else:
			f,ef,pos,epos = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(parallel_photutils)(cutouts[i],ecutouts[i],psfphot,init,True) for i in np.arange(len(cutouts))))
			pos['x_fit'] += xPix - size//2
			pos['y_fit'] += xPix - size//2
		
		if return_pos:
			return f, ef, pos, epos
		else:
			return f, ef





	def psf_photometry(self,xPix,yPix,size=7,snap='brightest',ext_shift=True,plot=False,diff=None,bkg_poly_order=3):
		"""
		Main PSF Photometry function

		Parameters
		----------
		xPix : float
			x pixel location for the initial guess of the target region.
		yPix : float
			y pixel location for the initial guess of the target region.
		size : int, optional
			Size of pixel cutout to use (should be odd). The default is 5.
		repFact : int, optional
			Super sampling factor for modelling. The default is 10.
		snap : str or int, optional
			Determines how psf position is fit. The default is 'brightest'.
			Valid Options:
				None = each frame's position will be fit and used when fitting for flux
				'brightest' = the position of the brightest cutout frame will be applied to all subsequent frames
				int = providing an integer allows for explicit choice of which frame to use as position reference
				'ref' = use the reference as the position fit point
		ext_shift : array_like, optional
			External shift in the pixel positions. The default is True.
		ext_shift : TYPE, optional
			DESCRIPTION. The default is True.
		plot : bool, optional
			Whether plots will shown. The default is False.
		diff : bool, optional
			If True then difference imaging will occur. The default is None.

		Returns
		-------
		flux : array_like
			Flux light curve across entire sector.

		"""
		
		from .psf_photom import create_psf

		if diff is None:
			diff = self.diff
		flux = []

		# if isinstance(xPix,(list,np.ndarray)):
		# 	self.moving_psf_phot()

		if type(snap) == str:
			if snap == 'all': 
				prf, cutouts, ecutouts = self._psf_initialise(size,(xPix,yPix),ref=(not diff))	# gather base PRF and the array of cutouts data
				inds = np.arange(len(cutouts))
				base = create_psf(prf,size)
				flux, eflux, pos = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_full)(cutouts[i],base,self.shift[i]) for i in inds))

				#prf, cutouts = self._psf_initialise(size,(xPix,yPix))	# gather base PRF and the array of cutouts data
				#xShifts = []
				#yShifts = []
				#for cutout in tqdm(cutouts):
				#	PSF = create_psf(prf,size)
				#	PSF.psf_position(cutout)
				#	PSF.psf_flux(cutout)
				#	flux.append(PSF.flux)
				#	yShifts.append(PSF.source_y)
				#	xShifts.append(PSF.source_x)
				#if plot:
				#	fig,ax = plt.subplots(ncols=3,figsize=(12,4))
				#	ax[0].plot(flux)
				#	ax[0].set_ylabel('Flux')
				#	ax[1].plot(xShifts,marker='.',linestyle=' ')
				#	ax[1].set_ylabel('xShift')
				#	ax[2].plot(yShifts,marker='.',linestyle=' ')
				#	ax[2].set_ylabel('yShift')
			else:
				if snap == 'brightest': # each cutout has position snapped to brightest frame fit position
					prf, cutouts, ecutouts = self._psf_initialise(size,(xPix,yPix),ref=(not diff))	# gather base PRF and the array of cutouts data
					bkg = self.bkg[:,int(yPix),int(xPix)]
					#lowbkg = bkg < np.nanpercentile(bkg,8)
					#weight = np.abs(np.nansum(cutouts[:,int(yPix)-1:int(yPix)+2,int(xPix)-1:int(xPix)+2],axis=(1,2))) / bkg
					weight = np.abs(np.nansum((cutouts / ecutouts)[:,int(yPix)-1:int(yPix)+2,int(xPix)-1:int(xPix)+2],axis=(1,2)))
					weight[np.isnan(weight)] = 0
					ind = np.argmax(weight)
					#ind = np.where(cutouts==np.nanmax(cutouts))[0][0]
					ref = cutouts[ind]
					base = create_psf(prf,size)
					base.psf_position(ref,ecutouts[ind],ext_shift=self.shift[ind])
				elif snap == 'ref':
					prf, cutouts,ecutouts = self._psf_initialise(size,(xPix,yPix),ref=True)	# gather base PRF and the array of cutouts data
					ref = cutouts[self.ref_ind]
					base = create_psf(prf,size)
					base.psf_position(ref,ecutouts[self.ref_ind])
					if diff:
						_, cutouts,ecutouts = self._psf_initialise(size,(xPix,yPix),ref=False)
				elif snap == 'fixed':
					prf, cutouts, ecutouts = self._psf_initialise(size,(xPix,yPix),ref=(not diff))	# gather base PRF and the array of cutouts data
					base = create_psf(prf,size)
				if self.parallel:
					inds = np.arange(len(cutouts))
					if self.delta_kernel is not None:
						flux, eflux = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_flux)(cutouts[i],ecutouts[i],base,self.shift[i],bkg_poly_order,self.delta_kernel[i]) for i in inds))
					else:
						flux, eflux = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_flux)(cutouts[i],ecutouts[i],base,self.shift[i],bkg_poly_order) for i in inds))
				else:
					for i in range(len(cutouts)):
						flux += [par_psf_flux(cutouts[i],ecutouts[i],base,self.shift[i])]
			if plot:
				plt.figure()
				plt.plot(flux)
				plt.ylabel('Flux')

			
		elif type(snap) == int:		# each cutout has position snapped to 'snap' frame fit position (snap is integer)
			base = create_psf(prf,size)
			base.psf_position(cutouts[snap])
			if self.parallel:
				inds = np.arange(len(cutouts))
				if self.delta_kernel is not None:
					flux, eflux = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_flux)(cutouts[i],ecutouts[i],base,self.shift[i],bkg_poly_order,self.delta_kernel[i]) for i in inds))
				else:
					flux, eflux = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(par_psf_flux)(cutouts[i],ecutouts[i],base,self.shift[i],bkg_poly_order) for i in inds))
			else:
				for i in range(len(cutouts)):
					flux += [par_psf_flux(cutouts[i],ecutouts[i],base,self.shift[i])]
			if plot:
				fig,ax = plt.subplots(ncols=1,figsize=(12,4))
				ax.plot(flux)
				ax.set_ylabel('Flux')
		flux = np.array(flux)
		eflux = np.array(eflux)
		return flux, eflux


	def orbit_ref_subtract(self):
		"""
		Subtract a per-orbit reference from each orbit's frames.

		Orbit assignments are obtained from TESSVectors if available, otherwise
		derived from time gaps > 0.5 days. For each orbit the reference is the
		median stack of that orbit's frames. A smoothed additive correction maps
		the primary orbit reference onto each secondary orbit before subtraction.

		Updates self.flux in place and stores self.orbit_segments.
		"""
		sector = self.sector
		camera = self.camera
		flux, segments, orbit_refs = orbit_ref_subtract(deepcopy(self.flux), self.mjd,
														sector=sector, camera=camera,
														vector_path=self._vector_path)
		self.flux = flux
		self.orbit_segments = segments
		self.orbit_refs = orbit_refs
		if self.verbose > 0:
			orbs, counts = np.unique(segments, return_counts=True)
			print(f'Orbit ref subtraction: {dict(zip(orbs, counts))} frames per orbit')

	def kernel_matching(self,size=7,diff=True):
		from .delta_function_fitting import parallel_delta_diff
		if diff:
			flux = deepcopy(self.flux + self.ref)
		else:
			flux = deepcopy(self.flux)
		mask = self.mask == 1 

		if self.parallel:
			d, kernel = zip(*Parallel(n_jobs=self.num_cores, backend=self.backend, verbose=self._joblib_verbose)(delayed(parallel_delta_diff)(frame,self.ref,mask,size) for frame in flux))
		else:
			d = []
			kernel = []
			for frame in flux:
				tmp = parallel_delta_diff(frame,self.ref,mask,size)
				d += [tmp[0]]
				kernel += [tmp[1]]
		d = np.array(d)
		self.delta_kernel = np.array(kernel)
		self.flux = d


	def reduce(self, aper = None, align = None, parallel = None, calibrate=None,
				bin_size = 0, plot = None, mask_scale = 1, ref_start=None, ref_stop=None,
				diff_lc = None,diff=None,verbose=None, tar_ap=3,sky_in=7,sky_out=11,
				moving_mask=None,mask=None,double_shift=False,corr_correction=None,test_seed=None,imaging=None):
		"""
		Reduce the images from the target pixel file and make a light curve with aperture photometry.
		This background subtraction method works well on tpfs > 50x50 pixels.

		Parameters
		----------
		aper : None, list or array_like, optional
			Aperature to do photometry on. The default is None.
		align : TYPE, optional
			DESCRIPTION. The default is None.
		parallel : bool, optional
			If True parallel processing will be used for background estimation and centroid shifts. The default is None.
		calibrate : TYPE, optional
			DESCRIPTION. The default is None.
		bin_size : int, optional
			If > 1 then the lightcurve will be binned by that amount. The default is 0.
		plot : TYPE, optional
			DESCRIPTION. The default is None.
		mask_scale : TYPE, optional
			DESCRIPTION. The default is 1.
		ref_start : TYPE, optional
			DESCRIPTION. The default is None.
		ref_stop : TYPE, optional
			DESCRIPTION. The default is None.
		diff_lc : TYPE, optional
			DESCRIPTION. The default is None.
		diff : TYPE, optional
			DESCRIPTION. The default is None.
		verbose : TYPE, optional
			DESCRIPTION. The default is None.
		tar_ap : TYPE, optional
			DESCRIPTION. The default is 3.
		sky_in : TYPE, optional
			DESCRIPTION. The default is 7.
		sky_out : TYPE, optional
			DESCRIPTION. The default is 11.
		moving_mask : TYPE, optional
			DESCRIPTION. The default is None.
		mask : TYPE, optional
			DESCRIPTION. The default is None.
		double_shift : TYPE, optional
			DESCRIPTION. The default is False.
		corr_correction : TYPE, optional
			DESCRIPTION. The default is None.
		test_seed : TYPE, optional
			DESCRIPTION. The default is None.

		Raises
		------
		ValueError
			DESCRIPTION.

		Returns
		-------
		None.

		"""
		# make reference
		try:
			_times = {}
			_t0_total = time.perf_counter()
			self._update_reduction_params(align, parallel, calibrate, plot, diff_lc, diff, verbose,corr_correction,imaging)

			if (self.flux.shape[1] < 30) & (self.flux.shape[2] < 30):
				small = True	
			else:
				small = False

			if small & self.align:
				print('Unlikely to get good shifts from a small tpf, so shift has been set to False')
				self.align = False

			_t = time.perf_counter()
			self.get_ref(ref_start,ref_stop)
			_times['reference frame'] = time.perf_counter() - _t
			if self.verbose > 0:
				print('made reference')
			# make source mask
			_t = time.perf_counter()
			if mask is None:
				self.make_mask(catalogue_path=self._catalogue_path,maglim=18,strapsize=7,scale=mask_scale)
				frac = np.nansum((self.mask == 0) * 1.) / (self.mask.shape[0] * self.mask.shape[1])
				#print('mask frac ',frac)
				if frac < 0.05:
					print('!!!WARNING!!! mask is too dense, lowering mask_scale to 0.5, and raising maglim to 15. Background quality will be reduced.')
					self.make_mask(catalogue_path=self._catalogue_path,maglim=15,strapsize=7,scale=0.2)
				if self.verbose > 0:
					print('made source mask')
			else:
				self.mask = mask
				if self.verbose > 0:
					print('assigned source mask')
			if moving_mask is not None:
					moving_mask = moving_mask > 0
					temp = np.zeros_like(self.flux,dtype=int)
					temp[:,:,:] = self.mask
					self.mask = temp | moving_mask
			_times['source mask'] = time.perf_counter() - _t
			# calculate background for each frame
			if self.verbose > 0:
				print('calculating background')
			
			# calculate the background
			#self.flux -= self.ref
			print('background pass 1...')
			_t = time.perf_counter()
			self.background(rerun_negative=True)
			self.flux -= self.bkg
			_times['background (pass 1)'] = time.perf_counter() - _t
			print('background pass 1 done')

			if np.isnan(self.bkg).all():
				raise ValueError('bkg all nans')
			
			# flux = strip_units(self.flux)
			# subtract background from unitless flux
			# self.flux = flux - self.bkg
			# get a ref with low background
			# self.ref = deepcopy(self.flux[self.ref_ind])
			if self.verbose > 0:
				print('background subtracted')
			
			
			if np.isnan(self.flux).all():
				raise ValueError('flux all nans')

			_t = time.perf_counter()
			if self.align and self.shift is None:
				if self.verbose > 0:
					print('aligning images')

				# try:
				if self._shift_method  == 'centroid':
					self.centroids_shifts_starfind()
				elif self._shift_method == 'difference':
					self.fit_shift(smooth=self._smooth_motion)
				elif self._shift_method == 'sep_core':
					from .sep_aligner import SepAligner
					aligner = SepAligner.from_tessreduce(self)
					aligner.run(verbose=self._joblib_verbose)
					if self._smooth_motion:
						aligner.smooth_shift(time=self.mjd,
											 gap_thresh=0.5, # days
											 update_shift=True)
					self.shift = aligner.shift
				else:
					m = f'Shift method {self._shift_method} is not supported, choose from:\ncentroid\ndifference\nsep_core'
					raise ValueError(m)
					#if double_shift:
					#self.shift_images()
					#self.ref = deepcopy(self.flux[self.ref_ind])

					#self.shift_images()

				# except:
				# 	print('Something went wrong, switching to serial')
				# 	self.parallel = False
				# 	if self._shift_method  == 'centroid':
				# 		self.centroids_shifts_starfind()
				# 	elif self._shift_method == 'minimize':
				# 		self.fit_shift()
			elif self.shift is None:
				self.shift = np.zeros((len(self.flux),2))

			_times['alignment'] = time.perf_counter() - _t
			print('alignment done')

			rawcube = self.rawflux if hasattr(self,'rawflux') else self.tpf.flux.value
			_bad_frames = np.nansum(rawcube,axis=(1,2))==0

			if not self.diff:
				if self.align:
					_t = time.perf_counter()
					self.shift_images()
					self.flux[_bad_frames] = np.nan
					_times['image shifting'] = time.perf_counter() - _t
					if self.verbose > 0:
						print('images shifted')
					#if self.kernel_match:
					#	self.kernel_matching(diff=False)

			if self.diff:
				if self.verbose > 0:
					print('!!Re-running for difference image!!')
				# reseting to do diffim
				_t = time.perf_counter()

				self.flux = rawcube
				self.flux = self.flux / self.qe

				if self.align:
					self.shift_images()

					if self.verbose > 0:
						print('shifting images')
				self._flux_aligned = deepcopy(self.flux)
				if test_seed is not None:
					self.flux += test_seed
				rawcube = self.rawflux if hasattr(self,'rawflux') else self.tpf.flux.value
				self.flux[_bad_frames] = np.nan
				# subtract reference
				if self._ref_type.lower() == 'single':
					self.ref = deepcopy(self.flux[self.ref_ind])
				elif self._ref_type.lower() == 'stack':
					self.stack_ref()
				self.ref -= np.nanpercentile(self.ref, 1)
				self.flux -= self.ref

				# self.ref -= self.bkg[self.ref_ind]
				self._ref_bkg = self.bkg[self.ref_ind]
				# remake mask
				self.make_mask(catalogue_path=self._catalogue_path,maglim=13,strapsize=7,scale=mask_scale*.5,useref=False)#Source_mask(ref,grid=0)
				frac = np.nansum((self.mask== 0) * 1.) / (self.mask.shape[0] * self.mask.shape[1])
				#print('mask frac ',frac)
				if frac < 0.05:
					print('!!!WARNING!!! mask is too dense, lowering mask_scale to 0.5, and raising maglim to 15. Background quality will be reduced.')
					self.make_mask(catalogue_path=self._catalogue_path,maglim=11,strapsize=7,scale=mask_scale*.5*.5)
				# assuming that the target is in the centre, so masking it out
				#m_tar = np.zeros_like(self.mask,dtype=int)
				#m_tar[self.ref.shape[0]//2,self.ref.shape[1]//2]= 1
				#m_tar = convolve(m_tar,np.ones((5,5)))
				#self.mask = self.mask | m_tar
				#mask
				#mask = convolve(self.mask,np.ones((3,3))) > 1
				#elf.mask = mask
				if moving_mask is not None:
					moving_mask = moving_mask > 0
					temp = np.zeros_like(self.flux,dtype=int)
					temp[:,:,:] = self.mask
					self.mask = temp | moving_mask
				_times['difference imaging setup'] = time.perf_counter() - _t
				print('diff setup done')

				self.bkg_orig = deepcopy(self.bkg)
				print('background pass 2...')
				_t = time.perf_counter()
				self.background(calc_qe = False,strap_iso = False,source_hunt=self._sourcehunt,
								gauss_smooth=self._bkg_gauss_sigma,interpolate=False,
								rerun_negative=False,rerun_diff=True,blend_dynamic=True)
				self.flux -= self.bkg
				_times['background (pass 2)'] = time.perf_counter() - _t
				print('background pass 2 done')

				if self.corr_correction:
					_t = time.perf_counter()
					print('correlation correction...')
					self.correlation_corrector()
					_times['correlation correction'] = time.perf_counter() - _t
					print('correlation correction done')
				if self.kernel_match:
					self.kernel_matching(diff=self.diff)
					if self.verbose > 0:
						print('kernels matched')

				if self.orbit_ref:
					if self.verbose > 0:
						print('orbit ref subtraction')
					self.orbit_ref_subtract()

			if self.calibrate:
				_t = time.perf_counter()
				print('field calibration')
				self.field_calibrate()
				_times['field calibration'] = time.perf_counter() - _t

			if self._create_lc:
				_t = time.perf_counter()
				self.lc, self.sky = self.diff_lc(plot=self.plot,diff=self.diff,tar_ap=tar_ap,sky_in=sky_in,sky_out=sky_out)
				_times['light curve'] = time.perf_counter() - _t

			if self.imaging:
				# if self.verbose > 0:
				# 	print('Retrieving external photometry')
				self.external_photometry()

			if self._timing:
				_total = time.perf_counter() - _t0_total
				print('\nReduce timing report:')
				for _name, _elapsed in _times.items():
					print(f'  {_name:<35s} {_elapsed:6.2f}s  ({100*_elapsed/_total:5.1f}%)')
				print(f'  {"total":<35s} {_total:6.2f}s')

		except Exception:
			print(traceback.format_exc())

		if self._cache_path is not None:
			try:
				os.remove(self._cache_path)
				if self.verbose > 0:
					print('Cache removed')
			except OSError:
				print(f'Failed to remove cache: {self._cache_path}')
			self._cache_path = None

		
	def external_photometry(self,size=50,phot=None):
		"""
		Perform aperture photometry on an external source

		Parameters
		----------
		size : int, optional
			Size of the cutout to use in arcseconds. The default is 50.
		phot : str, optional
			Type of photometry to choose, skymapper or Panstarrs. The default is None.
		"""

		event_cutout((self.ra,self.dec),size,phot)

	def make_lc(self,aperture = None,bin_size=0,zeropoint=None,scale='counts',clip = False):
		"""
		Perform aperature photometry on a time series of images

		Parameters
		----------
		aperture : array_like, optional
			An array of aperture sizes to use. The default is None.
		bin_size : int, optional
			Number of points to average. The default is 0.
		zeropoint : float, optional
			The calculated zeropoint of the data. The default is None.
		scale : bool, optional
			If True the light curve will be normalised to the median. The default is 'counts'.
			Valid options = [counts, magnitude, flux, normalise]
		clip : bool, optional
			Whether to clip the data. The default is False.

		Returns
		-------
		self.lc : array_like
			light curve for the pixels defined by the aperture

		"""
		
		# hack solution for new lightkurve
		flux = strip_units(self.flux)
		t = self.mjd

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
			 
		lc = Lightcurve(flux,aper)	#,scale = scale)
		if clip:
			mask = ~sigma_mask(lc)
			lc[mask] = np.nan
		if bin_size > 1:
			lc, t = self.bin_data(t,lc,bin_size)
		lc = np.array([t,lc])
		if (zeropoint is not None) & (scale=='mag'):
			lc[1,:] = -2.5*np.log10(lc[1,:]) + zeropoint
		self.lc = lc

	def lc_events(self,lc = None,err=None,duration=10,sig=5):
		"""
		Use clustering to detect individual high SNR events in a light curve.
		Clustering isn't incredibly robust, so it could be better.

		Parameters
		----------
		lc : array_like, optional
			lightcurve with the shape of (2,n), where the first index is time and the second is 
			flux. The default is None.
		err : array_like, optional
			Flux error to be used in weighting of events. The default is None.
		duration : int, optional
			How long an event needs to last for before being detected. The default is 10.
		sig : Float, optional
			Significance of the detection above the background. The default is 5.

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

		Parameters
		----------
		**kwargs : Various
			Keyword arguments, takes arguments for lightcurve events.

		Returns
		-------
		None.

		"""
		import matplotlib.pyplot as plt

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
		lc : array_like, optional
			lightcurve with the shape of (2,n), where the first index is time and the second is 
			flux. The default is None.
		err : array_like, optional
			Flux error to be used in weighting of events of size (n,). The default is None.
		Mask : array_like, optional
			1d one dimensional mask of the lightcurve to not be included in the detrending. The default is None.
		variable : bool, optional
			Determine whether the object is variable. The default is False.
		sig : float, optional
			Significance of the event before it gets excluded. The default is None.
		sig_up : Float, optional
			Upper sigma clip value . The default is 5.
		sig_low : Float, optional
			Lower sigma clip value. The default is 10.
		tail_length : str OR int, optional
			Option for setting the buffer zone of points after the peak. If it is 'auto' it 
			will be determined through functions, but if its an int then it will take the given 
			value as the buffer tail length for fine tuning. The default is ''.

		Raises
		------
		ValueError
			DESCRIPTION.

		Returns
		-------
		detrend : array_like
			Lightcurve with the stellar trends subtracted.

		"""
		import matplotlib.pyplot as plt

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
		lc : array_like, optional
			lightcurve with the shape of (2,n), where the first index is time and the second is 
			flux. The default is None.
		err : array_like, optional
			Flux error to be used in weighting of events of size (n,). The default is None.
		Mask : array_like, optional
			1d one dimensional mask of the lightcurve to not be included in the detrending. The default is None.
		variable : bool, optional
			Determine whether the object is variable. The default is False.
		sig : float, optional
			Significance of the event before it gets excluded. The default is None.
		sig_up : Float, optional
			Upper sigma clip value . The default is 5.
		sig_low : Float, optional
			Lower sigma clip value. The default is 10.
		tail_length : str OR int, optional
			Option for setting the buffer zone of points after the peak. If it is 'auto' it 
			will be determined through functions, but if its an int then it will take the given 
			value as the buffer tail length for fine tuning. The default is ''.

		Raises
		------
		ValueError
			"tail_length must be either 'auto' or an integer".

		Returns
		-------
		detrend : array_like
			Lightcurve with the stellar trends subtracted.

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
		# We now have a trend that should remove stellar variability, excluding flares.
		detrend = deepcopy(lc)
		detrend[1,:] = lc[1,:] - trends
		return detrend

	def bin_interp(self,lc=None,time_bin=6/24):
		"""
		Grabs the binned data and interpolates it to the original time values.

		Parameters
		----------
		lc : array_like, optional
			The lightcurve of the target in the form of three rows: mjd, flux, flux error. The default is None.	
		time_bin : float, optional
			The time (in days) for each bin to be for averaging data. The default is 6/24.

		Returns
		-------
		smooth : array_like
			Binned data in the form of three rows: mjd, flux, flux error.

		"""
		if lc is None:
			lc = self.lc
		if lc.shape[0] > lc.shape[1]:
			lc = lc.T
		binned = self.bin_data(lc=lc,time_bin=time_bin)
		finite = np.isfinite(binned[1])
		f1 = interp1d(binned[0,finite], binned[1,finite], kind='linear',fill_value='extrapolate')
		smooth = f1(lc[0])
		return smooth

	def detrend_star(self,lc=None,normalize=False):
		"""
		Removes trends, e.g. background or stellar variability from the lightcurve data.

		Parameters
		----------
		lc : array_like, optional
			The lightcurve of the target in the form of three rows: mjd, flux, flux error. The default is None.

		Returns
		-------
		detrended : array_like
			The lightcurve data where datapoints with strict gradient changes are ignored and a 
			savitsky-savgol filter is then applied to smooth out the data.

		"""
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
		if normalize:
			detrended[1] /= smooth
		else:
			detrended[1] -= smooth
		return detrended


	### serious calibration 
	def isolated_star_lcs(self):
		"""
		Serious calibration for isolated stars grabbing from either the skymapper or panstarrs database.
		This gives the lightcurves for the TESS target and the isolated stars in the field.

		Returns
		-------
		final_flux : array
			The final flux/TESS lightcurves for the stars in the final_d subset.
		final_d : array
			A subset of the skymapper or panstarrs observations/data tables that are not strongly 
			red in colour and below certain apparent magnitudes.

		"""
		if self.dec < -30:
			if self.verbose > 0:
				print('target is below -30 dec, calibrating to SkyMapper photometry.')
			table = Get_Catalogue(self.ra,self.dec,self.flux.shape,Catalog='skymapper')
			table = Skymapper_df(table)
			system = 'skymapper'
		else:
			if self.verbose > 0:
				print('target is above -30 dec, calibrating to PS1 photometry.')
			table = Get_Catalogue(self.ra,self.dec,self.flux.shape,Catalog='ps1')
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
			
			d.loc[i,'gmag'] = -2.5*np.log10(np.nansum(maselfflux(close.gmag.values,25))) + 25
			d.loc[i,'rmag'] = -2.5*np.log10(np.nansum(maselfflux(close.rmag.values,25))) + 25
			d.loc[i,'imag'] = -2.5*np.log10(np.nansum(maselfflux(close.imag.values,25))) + 25
			d.loc[i,'zmag'] = -2.5*np.log10(np.nansum(maselfflux(close.zmag.values,25))) + 25
			if system == 'ps1':
				d.loc[i,'ymag'] = -2.5*np.log10(np.nansum(maselfflux(close.ymag.values,25))) + 25
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
			mask[int(np.round(d.row.values[i],0)),int(np.round(d.col.values[i],0))] = 1
			mask = convolve(mask,np.ones((3,3)))
			flux += [np.nansum(tflux*mask,axis=(1,2))]
			m2 = np.zeros_like(self.ref)
			m2[int(np.round(d.row.values[i],0)),int(np.round(d.col.values[i]))] = 1
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

		final_d = d.iloc[eind]
		final_flux = flux[eind]

		return final_flux, final_d

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

		Parameters
		----------
		zp_single : bool, optional
			valid options are True or False. The default is True.		
				if True all points through time are averaged to a single zp
				if False then the zp is time varying, creating an extra photometric correction
				for light curves, but with increased error in the zp.
		plot : bool, optional
			If True then diagnostic plots will be created. The default is None.
		savename : str, optional
			The name used for the saving files. The default is None.

		Returns
		-------
		None.
			self.ebv : float 
				Estimated E(B-V) extinction from stellar locus regression
			self.zp/tzp : float
				TESS photometric zeropoint
			self.zp_e/tzp_e : float
				Error in the photometric zeropoint

		"""
		import matplotlib.pyplot as plt

		if plot is None:
			plot = self.diagnostic_plot
		if savename is None:
			savename = self.savename
		if self.dec < -30:
			if self.verbose > 0:
				print('target is below -30 dec, calibrating to SkyMapper photometry.')
			table = Get_Catalogue(self.ra,self.dec,self.flux.shape,Catalog='skymapper')
			system = 'skymapper'
			if table is None:
				print('WARNING: SkyMapper unavailable, skipping field calibration.')
				return
		else:
			if self.verbose > 0:
				print('target is above -30 dec, calibrating to PS1 photometry.')
			table = Get_Catalogue(self.ra,self.dec,self.flux.shape,Catalog='ps1')
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
			

		ind = (table.imag.values < 19) & (table.imag.values > 13.5)
		tab = table.iloc[ind]
		
		e, dat = Tonry_reduce(tab,plot=plot,savename=savename,system=system)
		self.ebv = e[0]

		gr = (dat.gmag - dat['rmag']).values
		ind = (gr < 1) & (dat['imag'].values < 8)
		d = dat.iloc[ind]
		
		x,y = self.wcs.all_world2pix(d.RAJ2000.values,d.DEJ2000.values,0)
		d['col'] = x
		d['row'] = y
		pos_ind = (-5 < x) & (x < self.ref.shape[1]+5) & (-5 < y) & (y < self.ref.shape[0]+5)
		d = d.iloc[pos_ind]
		x = x[pos_ind]; y = y[pos_ind]
		

		# account for crowding 
		for i in range(len(d)):
			xx = d.col.values[i]
			yy = d.row.values[i]
			
			dist = np.sqrt((tab['col'].values-xx)**2 + (tab['row'].values-yy)**2)
			
			ind = dist < 1
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

		
		maglim = 8.5
		maglim_bright = 10 
		dist = np.sqrt((x[:,np.newaxis] - x[np.newaxis,:])**2 + (y[:,np.newaxis] - y[np.newaxis,:])**2)
		mag_diff = d['tmag'].values[:,np.newaxis] - d['tmag'].values[np.newaxis,:]
		mag_diff[mag_diff == 0] = np.nan
		mag_diff[mag_diff < -2] = np.nan
		mag_diff = np.isnan(mag_diff)
		dist[dist==0] = np.nan
		dist[mag_diff] = np.nan
		min_dist = np.nanmin(dist,axis=1)
		ind = min_dist > 5
		if sum(ind) < 10:
			ind = min_dist > 3
		d = d.iloc[ind]
		ind2 = (d['tmag'].values < maglim) & (d['tmag'].values > maglim_bright)
		d = d.iloc[ind2]
		xx = x[ind][ind2]; yy = y[ind][ind2]
		ind = (xx > 5) & (xx < self.ref.shape[1]-5) & (yy > 5) & (yy < self.ref.shape[0]-5)
		d = d.iloc[ind]
		xx = xx[ind]; yy = yy[ind]
		d['col'] = xx; d['row'] = yy

		#self.cat['cal'] = 0
		#self.cat['cal'].iloc[ind] = 1

		if len(d) == 0:
			print('!!! No suitable calibration sources !!!\nSetting to the default value of zp=20.44')
			self.zp = 20.44
			self.zp_e = 0.05
			# backup for when messing around with flux later
			self.tzp = 20.44
			self.tzp_e = 0.05
			return

		flux = []
		eflux = []
		eind = np.zeros(len(d))
		for i in range(len(d)):
			mask = np.zeros_like(self.ref)
			xx = int(np.round(d['col'].values[i],0)); yy = int(np.round(d['row'].values[i]))
			mask[yy,xx] = 1
			mask = convolve(mask,np.ones((3,3)))
			if self.phot_method == 'aperture':
				flux += [np.nansum(tflux*mask,axis=(1,2))]
			elif self.phot_method == 'psf':
				f, e = self.psf_photometry(xPix=xx,yPix=yy,snap='ref',diff=False)
				flux += [f]
				eflux += [e]
			m2 = np.zeros_like(self.ref)
			m2[int(np.round(d['row'].values[i],0)),int(np.round(d['col'].values[i]))] = 1
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
		zp = d['tmag'].values[:,np.newaxis] + 2.5*np.log10(flux) 
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
			plt.imshow(ref,origin='lower',vmax = np.percentile(ref[nonan],90),vmin=np.percentile(ref[nonan],10))
			cbar = plt.colorbar()
			cbar.ax.set_ylabel('Counts',fontsize=12)
			plt.plot(d.col.iloc[eind],d.row.iloc[eind],'rx')
			plt.title('Calibration sources')
			plt.ylabel('Row',fontsize=15)
			plt.xlabel('Column',fontsize=15)
			
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
			plt.plot(self.mjd[mask],mzp[mask],'.',alpha=0.5)
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
			mzp,stdzp = smooth_zp(zp, self.mjd)
			compare = (abs(mzp-20.44) > 2).any()

		if compare:
			print('!!!WARNING!!! field calibration is unreliable, using the default zp = 20.44')
			self.zp = 20.44
			self.zp_e = 0.05
			# backup for when messing around with flux later
			self.tzp = 20.44
			self.tzp_e = 0.05
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

		Parameters
		----------
		zp : float, optional
			Zeropoint to use for conversion. If None, use the default zp from the object. The default is None.
		zp_e : float, optional
			Error on the zeropoint to use for conversion. If None, use the default zp_e from the object. The default is 0.

		Returns
		-------
		lc : array_like
			Lightcurve in magnitude space. mjd, magnitude, magnitude_error.

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

		Parameters
		----------
		zp : float, optional
			TESS zeropoint. The default is None.
		zp_e : float, optional
			Error in the TESS zeropoint. The default is 0.
		flux_type : str, optional
			The units for the flux output. The default is 'mjy'.
			Valid options:
				mjy 
				jy
				erg/cgs
				tess/counts
		plot : bool, optional
			Plot the field calibration figures, if used. The default is False.

		Raises
		------
		ValueError
			Flux Type is not a valid option, please choose from:\njy\nmjy\ncgs/erg\ntess/counts'.

		Returns
		-------
		None.
			self.lc : array_like
				Returns the lightcurve in flux units, with mjd, flux, flux error being the three rows.
			self.zp : float
				Returns the zeropoint used for the magnitude conversion.
			self.zp_e : float 
				Returns the error on the zeropoint used for the magnitude conversion.
			self.lc_units : str
				Returns the flux units of the lightcurve.

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
			flux_zp = 8.4
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
			self.zp = self.zp * 0 + 8.4
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