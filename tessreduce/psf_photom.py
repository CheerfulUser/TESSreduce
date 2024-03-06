import numpy as np
import PRF
from skimage.util.shape import view_as_windows
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.ndimage import shift
"""
PSF photometry class repurposed from Starkiller by Hugh Roxburgh
https://github.com/CheerfulUser/starkiller

"""

def downSample2d(arr,sf):

    isf2 = 1.0/(sf*sf)    # scalefactor
    windows = view_as_windows(arr, (sf,sf), step = sf)  # automatically scale down
    return windows.sum(3).sum(2)*isf2

class create_psf():
    def __init__(self,prf,size,repFact=10):
        
        self.prf = prf
        self.size = size
        self.source_x = 0         # offset of source from centre
        self.source_y = 0         # offset of source from centre
        self.repFact=repFact     # supersample multiplication

        # -- Finds centre of kernel -- #
        self.cent=self.size/2.-0.5

        self.psf = None

    def source(self,shiftx=0,shifty=0,ext_shift=[0,0]):
        
        centx_s = self.cent + shiftx    # source centre
        centy_s = self.cent + shifty

        psf = self.prf.locate(centx_s-ext_shift[1],centy_s-ext_shift[0], (self.size,self.size))
        psf = shift(psf,ext_shift)
        self.psf = psf/np.nansum(psf)

    def minimize_position(self,coeff,image,ext_shift):

        self.source_x = coeff[0]
        self.source_y = coeff[1]
        
        # -- generate psf -- #
        self.source(shiftx = self.source_x, shifty = self.source_y,ext_shift=ext_shift)

        # -- calculate residuals -- #
        diff = abs(image - self.psf)
        residual = np.nansum(diff)
        return np.exp(residual)
    
    def psf_position(self,image,limx=1,limy=1,ext_shift=[0,0]):
        """
        Fit the PSF. Limx,y dictates bounds for position of the source
        """

        normimage = image / np.nansum(image)    # normalise the image
        coeff = [self.source_x,self.source_y]
        lims = [[-limx,limx],[-limy,limy]]
        
        # -- Optimize -- #
        res = minimize(self.minimize_position, coeff, args=(normimage,ext_shift), method='Powell',bounds=lims)
        self.psf_fit = res

    def minimize_psf_flux(self,coeff,image):
        res = np.nansum(abs(image - self.psf*coeff[0]))
        return res

    def psf_flux(self,image,ext_shift=None):
        if self.psf is None:
            self.source(shiftx=self.source_x,shifty=self.source_y)
        if ext_shift is not None:
            self.source(ext_shift=ext_shift)
        mask = np.zeros_like(self.psf)
        mask[self.psf > np.nanpercentile(self.psf,70)] = 1
        f0 = np.nansum(image*mask)
        bkg = np.nanmedian(image[~mask.astype(bool)])
        image = image - bkg
        
        #f0 = np.nansum(image)

        res = minimize(self.minimize_psf_flux,f0,args=(image),method='Nelder-Mead')
        self.flux = res.x[0]
        self.image_residual = image - self.psf*self.flux