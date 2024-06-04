import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from scipy.signal import fftconvolve
import os

package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'
from .catalog_tools import *
from .helpers import *

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def size_limit(x,y,image):
    yy,xx = image.shape
    ind = ((y > 0) & (y < yy-1) & (x > 0) & (x < xx-1))
    return ind


def circle_app(rad):
    """
    makes a kinda circular aperture, probably not worth using.
    """
    mask = np.zeros((int(rad*2+.5)+1,int(rad*2+.5)+1))
    c = rad
    x,y =np.where(mask==0)
    dist = np.sqrt((x-c)**2 + (y-c)**2)

    ind = (dist) < rad + .2
    mask[y[ind],x[ind]]= 1
    return mask


def ps1_auto_mask(table,Image,scale=1):
    """
    Make a source mask using the PS1 catalog
    """
    image = np.zeros_like(Image)
    x = table.x.values
    y = table.y.values
    x = (x+.5).astype(int)
    y = (y+.5).astype(int)
    m = table.mag.values
    ind = size_limit(x,y,image)
    x = x[ind]; y = y[ind]; m = m[ind]
    
    magim = image.copy()
    magim[y,x] = m
    
    masks = {}
    
    mags = [[18,17],[17,16],[16,15],[15,14],[14,13.5],[13.5,12]]
    size = (np.array([3,4,5,6,7,8]) * scale).astype(int)
    for i in range(len(mags)):
        m = ((magim > mags[i][1]) & (magim <= mags[i][0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mags[i][0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks

def gaia_auto_mask(table,Image,scale=1):
    """
    Make a source mask from gaia source catalogue
    """
    image = np.zeros_like(Image)
    x = table.x.values
    y = table.y.values
    x = (x+.5).astype(int)
    y = (y+.5).astype(int)
    m = table.mag.values
    ind = size_limit(x,y,image)
    x = x[ind]; y = y[ind]; m = m[ind]
    
    maglim = np.zeros_like(image,dtype=float)
    magim = image.copy()
    magim[y,x] = m
    
    masks = {}
    
    mags = [[18,17],[17,16],[16,15],[15,14],[14,13.5],[13.5,12],[12,10],[10,9],[9,8],[8,7]]
    size = (np.array([3,4,5,6,7,8,10,14,16,18])*scale).astype(int)
    for i in range(len(mags)):
        m = ((magim > mags[i][1]) & (magim <= mags[i][0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mags[i][0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks
    
def Big_sat(table,Image,scale=1):
    """
    Make crude cross masks for the TESS saturated sources.
    The properties in the mask need some fine tuning.
    """
    image = np.zeros_like(Image)
    i = (table.mag.values < 7) #& (gaia.gaia.values > 2)
    sat = table.iloc[i]
    x = sat.x.values
    y = sat.y.values
    x = (x+.5).astype(int)
    y = (y+.5).astype(int)
    m = sat.mag.values
    ind = size_limit(x,y,image)
    
    x = x[ind]; y = y[ind]; m = m[ind]
    
    
    satmasks = []
    for i in range(len(x)):
        mag = m[i]
        mask = np.zeros_like(image,dtype=float)
        if (mag <= 7) & (mag > 5):
            body   = int(13 * scale)
            length = int(20 * scale)
            width  = int(3 * scale)
        if (mag <= 5) & (mag > 4):
            body   = 15 * scale
            length = int(60 * scale)
            width  = int(5 * scale)
        if (mag <= 4):# & (mag > 4):
            body   = int(22 * scale)
            length = int(115 * scale)
            width  = int(7 * scale)
        body = int(body) # no idea why this is needed, but it apparently is.
        kernel = np.zeros((body*2+1,body*2+1))
        yy,xx = np.where(kernel == 0)
        dist = np.sqrt((yy-body)**2 + (xx-body)**2)
        ind = dist <= body+1
        kernel[yy[ind],xx[ind]] = 1
        mask[y[i],x[i]] = 1 
        conv = fftconvolve(mask, kernel,mode='same')#.astype(int)
        mask = (conv >.1) * 1.
        
        ylow = y[i]-length; yhigh=y[i]+length
        xlow = x[i]-width; xhigh = x[i]+width
        if ylow < 0:
            ylow = 0
        if xlow < 0:
            xlow = 0
        mask[ylow:yhigh,xlow:xhigh] = 1 
        ylow = y[i]-width
        xlow = x[i]-length
        if ylow < 0:
            ylow = 0
        if xlow < 0:
            xlow = 0
        mask[ylow:y[i]+width,xlow:x[i]+length] = 1 
        
        satmasks += [mask]
    satmasks = np.array(satmasks)
    return satmasks

def Strap_mask(Image,col,size=4):
    strap_mask = np.zeros_like(Image)
    straps = pd.read_csv(package_directory + 'tess_straps.csv')['Column'].values - col + 44
    strap_in_tpf = straps[((straps > 0) & (straps < Image.shape[1]))]
    strap_mask[:,strap_in_tpf] = 1
    big_strap = fftconvolve(strap_mask,np.ones((size,size)),mode='same') > .5
    return big_strap

def Cat_mask(tpf,catalogue_path=None,maglim=19,scale=1,strapsize=3,badpix=None,ref=None,sigma=3):

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

	if catalogue_path is not None:
		gaia  = external_load_cat(catalogue_path,maglim)
		coords = tpf.wcs.all_world2pix(gaia['ra'],gaia['dec'], 0)
		gaia['x'] = coords[0]
		gaia['y'] = coords[1]
	else:
		gp,gm = Get_Gaia(tpf,magnitude_limit=maglim)
		gaia  = pd.DataFrame(np.array([gp[:,0],gp[:,1],gm]).T,columns=['x','y','mag'])

	image = tpf.flux[10]
	image = strip_units(image)

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