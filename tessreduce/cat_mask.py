import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
import os

from .catalog_tools import Get_Gaia, external_load_cat
from .helpers import *

package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

# turn off runtime warnings (lots from logic on nans)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def size_limit(x,y,image):
    """
    Find indices where pixels are inside size of image.
    """
    yy,xx = image.shape
    ind = ((y > 0) & (y < yy-1) & (x > 0) & (x < xx-1))
    return ind


def circle_app(rad):
    """
    Makes a kinda circular aperture, probably not worth using.
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
    Make a source mask using the PS1 catalog.
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
    for i, mag in enumerate(mags):
        m = ((magim > mag[1]) & (magim <= mag[0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mag[0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks

def gaia_auto_mask(table,Image,scale=1):
    """
    Make a source mask from gaia source catalogue.

    Parameters
    ----------

    table : Dataframe like
        Table of sources from gaia.
    Image : ArrayLike
        Flux of a reference image.
    scale : int, optional 
        Scale factor for size of kernel mask. The default is 1.

    Returns
    -------

    masks : dict
        Source mask for gaia sources. 

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
    
    mags = [[18,17],[17,16],[16,15],[15,14],[14,13.5],[13.5,12],[12,10],[10,9],[9,8],[8,7]]
    size = (np.array([3,4,5,6,7,8,10,14,16,18])*scale).astype(int)
    for i, mag in enumerate(mags):
        m = ((magim > mag[1]) & (magim <= mag[0])) * 1.
        k = np.ones((size[i],size[i]))
        conv = fftconvolve(m, k,mode='same')#.astype(int)
        masks[str(mag[0])] = (conv >.1) * 1.
    masks['all'] = np.zeros_like(image,dtype=float)
    for key in masks:
        masks['all'] += masks[key]
    masks['all'] = (masks['all'] > .1) * 1.
    return masks
    
def Big_sat(table,Image,scale=1):
    """
    Make crude cross masks for the TESS saturated sources.
    The properties in the mask need some fine tuning.

    Parameters
    ----------

    table : Dataframe like
        Table of sources from gaia.
    Image : ArrayLike
        Flux of a reference image.
    scale : int, optional 
        Scale factor for size of kernel mask. The default is 1.

    Returns
    -------

    masks : array
        Source mask for saturated sources. 

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
    """
    Make a mask for the electrical straps on TESS CCDs.

    Parameters
    ----------

    Image : ArrayLike
        Flux of a reference image.
    col : int
        Reference index of TPF columns.
    size : int, optional
        Width of the strap in pixels. The default is 4.
    """

    strap_mask = np.zeros_like(Image)
    straps = pd.read_csv(package_directory + 'tess_straps.csv')['Column'].values - col + 44
    strap_in_tpf = straps[((straps > 0) & (straps < Image.shape[1]))]
    strap_mask[:,strap_in_tpf] = 1
    big_strap = fftconvolve(strap_mask,np.ones((size,size)),mode='same') > .5
    return big_strap

def Cat_mask(tpf,catalogue_path=None,maglim=19,scale=1,strapsize=3,ref=None,sigma=3):

	"""
	Make a source mask from the PS1 and Gaia catalogs.

	Parameters
	----------
	tpf : lightkurve target pixel file
		tpf of the desired region
    catalogue_path : str, optional
		Local path to source catalogue if using TESSreduce in offline mode. The default is None.
	maglim : float, optional
		Magnitude limit in PS1 i band and Gaia G band for sources to include in source mask. The default is 19.
    scale : float, optional
		Scale factor for default mask size. The default is 1. 
	strapsize : float, optional
        Width in pixels of the mask for TESS' electrical straps. The default is 6.  

	Returns
	-------
	total mask : bitmask
		a bitwise mask for the given tpf. Bits are as follows:
            0 - background
            1 - catalogue source
            2 - saturated source
            4 - strap mask
            8 - bad pixel (not used)
    gaia : DataframeLike
        Gaia catalogue of sources included in mask.

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
	
	if strapsize > 0: 
		strap = Strap_mask(image,tpf.column,strapsize).astype(int) * 4 # assign 4 bit 
	else:
		strap = np.zeros_like(image,dtype=int)

	totalmask = mask | sat | strap
	
	return totalmask, gaia