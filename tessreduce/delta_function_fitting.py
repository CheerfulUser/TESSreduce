from scipy.optimize import minimize
from scipy import signal
from astropy.convolution import Gaussian2DKernel
import numpy as np

from scipy.optimize import minimize

def Delta_basis(Size = 11):
    kernel = np.zeros((Size,Size))
    x,y = np.where(kernel==0)
    middle = int(len(x)/2)
    basis = []
    for i in range(len(x)):
        b = kernel.copy()
        if (x[i] == x[middle]) & (y[i] == y[middle]):
            b[x[i],y[i]] = 1
        else:
            b[x[i],y[i]] = 1
            b[x[middle],y[middle]] = -1
        basis += [b]
    basis = np.array(basis)
    coeff = np.ones(len(basis))
    return basis, coeff

def Delta_kernel(reference,image,Size=11,mask=None):
    if mask is None:
        mask = np.ones_like(image)
    mask[mask == 0] = np.nan
    Basis, coeff_0 = Delta_basis(Size)
    bds = []
    for i in range(len(coeff_0)):
        bds += [(0,1)]
    coeff_0 *= 0.01
    coeff_0[Size//2+1] = 0.95
    res = minimize(optimize_delta, coeff_0, args=(Basis,reference,image,Size,mask),
                   bounds=bds,method='Powell')
    k = np.nansum(res.x[:,np.newaxis,np.newaxis]*Basis,axis=0)
    return k
        
def optimize_delta(Coeff, Basis, reference, image,size,mask):
    kernel = np.nansum(Coeff[:,np.newaxis,np.newaxis]*Basis,axis=0)

    template = signal.fftconvolve(reference, kernel, mode='same')
    
    im = image.copy()
    
    res = np.nansum(abs(im[size//2:-size//2,size//2:-size//2]-template[size//2:-size//2,size//2:-size//2])*mask[size//2:-size//2,size//2:-size//2])
    #print(res)
    return res


def parallel_delta_diff(image,reference,mask=None,size=11):
    kernel = Delta_kernel(reference,image,Size=11,mask=mask)
    template = signal.fftconvolve(reference, kernel, mode='same')
    diff = image - template
    return diff, kernel


    