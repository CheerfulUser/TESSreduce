import numpy as np
from skimage.util.shape import view_as_windows
from scipy.optimize import minimize
from scipy.ndimage import shift
from scipy.signal import fftconvolve

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

def polynomial_surface(x, y, coeffs, order=2):
    """Evaluate an n-order polynomial surface."""
    z = np.zeros_like(x,dtype=float)
    ind = 0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            z += coeffs[ind] * (x ** i) * (y ** j)
            ind += 1
    return z


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

    def minimize_position(self,coeff,image,error,ext_shift):#,surface=True,order=2):
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
        # if surface:
        #     x = np.arange(image.shape[1])
        #     y = np.arange(image.shape[0])
        #     yy,xx = np.meshgrid(y,x)
        #     plane_coeff = coeff[2:]
        #     s = polynomial_surface(xx,yy,plane_coeff,order)
        # else:
        #     s = 0

        self.source_x = coeff[0]
        self.source_y = coeff[1]
        
        # -- generate psf -- #
        self.source(shiftx = self.source_x, shifty = self.source_y,ext_shift=ext_shift)

        # -- calculate residuals -- #
        diff = abs(image - self.psf)**2
        residual = np.nansum(diff/error)
        return residual#np.exp(residual)
    
    def psf_position(self,image,error,limx=0.8,limy=0.8,ext_shift=[0,0]):#,surface=False,order=2):
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
        if error is None:
            error = np.ones_like(image)
        #brightloc = 
        if np.nansum(image) > 0:
            if np.isfinite(ext_shift).all():
                ext_shift[0] = 0; ext_shift[1] = 0
            normimage = image / np.nansum(image)    # normalise the image

            # if surface:
            #     num_coeffs = (poly_order + 1) * (poly_order + 2) // 2
            #     coeff = np.zeros(num_coeffs + 2)
            #     coeff[0] = self.source_x; coeff[1] = self.source_y
            #     lims = [[-limx,limx],[-limy,limy]]
            #     for i in range(num_coeffs):
            #         lims += [-np.inf,np.inf]
            # else:
            coeff = [self.source_x,self.source_y]
            lims = [[-limx,limx],[-limy,limy]]
            
            # -- Optimize -- #
            res = minimize(self.minimize_position, coeff, args=(normimage,error,ext_shift), method='Powell',bounds=lims)
            print('Optimal PSF shift: ', res.x)
            self.psf_fit = res
        else:
            self.psf_fit = None

    def minimize_psf_flux(self,coeff,image,error=None,surface=True,order=2,kernel=None):

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
        if surface:
            x = np.arange(image.shape[1])
            y = np.arange(image.shape[0])
            yy,xx = np.meshgrid(y,x)
            plane_coeff = coeff[1:]
            s = polynomial_surface(xx,yy,plane_coeff,order)
        else:
            s = 0
        if kernel is not None:
            self.psf = fftconvolve(self.psf, kernel, mode='same')

        res = np.nansum((image - self.psf*coeff[0] - s)**2/error)
        return res

    def psf_flux(self,image,error=None,ext_shift=None,surface=True,poly_order=3,kernel=None):

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
        if error is None:
            error = np.ones_like(image)
        if self.psf is None:
            self.source(shiftx=self.source_x,shifty=self.source_y)
        if (ext_shift is not None) & np.isfinite(ext_shift).all():
            self.source(ext_shift=ext_shift)
        mask = np.zeros_like(self.psf)
        mask[self.psf > np.nanpercentile(self.psf,90)] = 1
        f0 = np.nansum(image*mask)
        #bkg = np.nanmedian(image[~mask.astype(bool)])
        #image = image - bkg

        if surface:
            num_coeffs = (poly_order + 1) * (poly_order + 2) // 2
            initial = np.zeros(num_coeffs + 1)
            initial[0] = f0
        else:
            initial = f0
        
        res = minimize(self.minimize_psf_flux,initial,args=(image,error,surface,poly_order,kernel),method='BFGS')
        #res = minimize(self.minimize_psf_flux,initial,args=(image,error,surface,poly_order,kernel),method='Powell')
        error = np.sqrt(np.diag(res['hess_inv']))

        self.res = res
        self.flux = res.x[0]
        self.eflux = error[0]
        
        if surface:
            x = np.arange(image.shape[1])
            y = np.arange(image.shape[0])
            yy,xx = np.meshgrid(y,x)
            plane_coeff = res.x[1:]
            s = polynomial_surface(xx,yy,plane_coeff,poly_order)
        else:
            s = image * 0
        self.surface = s
        self.image_residual = image - self.psf*self.flux - s
