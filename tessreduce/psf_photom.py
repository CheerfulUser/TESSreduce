import numpy as np
from skimage.util.shape import view_as_windows
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.ndimage import shift
"""
PSF photometry class repurposed from Starkiller by Hugh Roxburgh
https://github.com/CheerfulUser/starkiller

"""

def downSample2d(arr,sf):
        """

        ----
        Inputs
        ----

        arr : array
            array to scale to 2d
        sf : float
            dimension of the scaled array
            
        ----
        Output
        ----

        array:
            rescaled array in 2d

        """
        
        isf2 = 1.0/(sf*sf)    # scalefactor
        windows = view_as_windows(arr, (sf,sf), step = sf)  # automatically scale down
        return windows.sum(3).sum(2)*isf2

class create_psf():
    def __init__(self,prf,size):
        
        self.prf = prf
        self.size = size
        self.source_x = 0         # offset of source from centre
        self.source_y = 0         # offset of source from centre

        # -- Finds centre of kernel -- #
        self.cent=self.size/2.-0.5

        self.psf = None

    def source(self,shiftx=0,shifty=0,ext_shift=[0,0]):

        """
        Creates the psf of a given source
        
        ----
        Inputs
        ----

        shiftx : float
            shift of source x-coordinate from centre of kernel
        shifty : float
            shift of source y-coordinate from centre of kernel
        ext_shift : array
            shift vector for psf array corresponding to offset from kernel centre to source centre
            
        ----
        Output
        ----
        
        psf: array
            psf for source at the given coordinates relative to the kernel centre

        """
        
        centx_s = self.cent + shiftx    # source centre
        centy_s = self.cent + shifty

        psf = self.prf.locate(centx_s-ext_shift[1],centy_s-ext_shift[0], (self.size,self.size))
        psf = shift(psf,ext_shift)
        self.psf = psf/np.nansum(psf)

    def minimize_position(self,coeff,image,ext_shift):
        """
        Applies an exponential function using psf residuals to optimise the psf fit.
        
        ----
        Inputs
        ----

        coeff : array
            offset of source from centre
        image : array
            actual flux array
        ext_shift : array
            shift vector for psf array corresponding to offset from kernel centre to source centre
            
        ----
        Output
        ----
        
        array:
            optimization model used for psf fitting
            
        """
        
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
        Finds the optimal psf fit
        
        ----
        Inputs
        ----

        image : array
            flux array to fit psf to
        limx : float
            bound to psf fit in (+ and -) x-direction
        limy : float
            bound to psf fit in (+ and -) Y-direction
        ext_shift : array
            shift vector for psf array corresponding to offset from kernel centre to source centre

        ----
        Outputs
        ----
        psf_fit: array
            Optimal psf fit to input image
            
        """

        normimage = image / np.nansum(image)    # normalise the image
        coeff = [self.source_x,self.source_y]
        lims = [[-limx,limx],[-limy,limy]]
        
        # -- Optimize -- #
        res = minimize(self.minimize_position, coeff, args=(normimage,ext_shift), method='Powell',bounds=lims)
        self.psf_fit = res

    def minimize_psf_flux(self,coeff,image):

        """
        
        Calculates residuals to optimise psf flux
        
        ----
        Inputs
        ----

        coeff : array
            source mask for psf 
        image : array
            image flux array

        ----
        Outputs
        ----

        res: float
            flux residual of psf relative to image
        
        """
        res = np.nansum(abs(image - self.psf*coeff[0]))
        return res

    def psf_flux(self,image,ext_shift=None):

        """
        
        Finds the optimal flux fit 

        ----
        Inputs
        ----

        image : array
            flux array to fit psf to
        ext_shift : array
            shift vector for psf array corresponding to offset from kernel centre to source centre

        ----
        Outputs
        ----
        
        flux : array
            optimal fit of psf flux to image
        image_residual : array
            residual of optimal psf flux fit 
            
        """
        
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
