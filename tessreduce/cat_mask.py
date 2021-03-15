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
            width  = int(4 * scale)
        if (mag <= 5) & (mag > 4):
            body   = 15 * scale
            length = int(60 * scale)
            width  = int(10 * scale)
        if (mag <= 4):# & (mag > 4):
            body   = int(25 * scale)
            length = int(115 * scale)
            width  = int(10 * scale)
        body = int(body) # no idea why this is needed, but it apparently is.
        kernal = np.ones((body*2,body*2))
        mask[y[i],x[i]] = 1 
        conv = fftconvolve(mask, kernal,mode='same')#.astype(int)
        mask = (conv >.1) * 1.

        mask[y[i]-length:y[i]+length,x[i]-width:x[i]+width] = 1 
        mask[y[i]-width:y[i]+width,x[i]-length:x[i]+length] = 1 
        
        satmasks += [mask]
    satmasks = np.array(satmasks)
    return satmasks

def Strap_mask(Image,col,size=3):
    strap_mask = np.zeros_like(Image)
    path = '/user/rridden/feet/'
    straps = pd.read_csv(package_directory + 'tess_straps.csv')['Column'].values - col + 44
    strap_in_tpf = straps[((straps > 0) & (straps < Image.shape[1]))]
    strap_mask[:,strap_in_tpf] = 1
    big_strap = fftconvolve(strap_mask,np.ones((size,size)),mode='same') > .5
    return big_strap
