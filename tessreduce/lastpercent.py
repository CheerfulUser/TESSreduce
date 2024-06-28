import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from copy import deepcopy

def cor_minimizer(coeff,pix_lc,bkg_lc):
    """
    Calculates the Pearson r correlation coefficent between the background subtracted lightcurve and the background itself. Takes inputs in a form for minimizing methods to be run on this function.

    Parameters:
    ----------
    coeff: float
        The multiplier on the background flux to be subtracted from the lightcurve. This is the variable being changed in any minimization.
    pix_lc: ArrayLike
        The full lightcurve of pixel flux data. Has a back 
    bkg_lc: ArrayLike
        The background lightcurve, to be multiplied by coeff and subtracted from pix_lc
    
    Returns:
    -------
    corr: float
        The absolute value of the Pearson r correlation coefficent between the background subtracted lightcurve and the background. 
    """
    lc = pix_lc - coeff * bkg_lc
    ind = np.isfinite(lc) & np.isfinite(bkg_lc)
    #bkgnorm = bkg_lc/np.nanmax(bkg_lc)
    #pixnorm= (lc - np.nanmedian(lc))
    #pixnorm = pixnorm / np.nanmax(abs(pixnorm))
    corr = pearsonr(lc[ind],bkg_lc[ind])[0]
    return abs(corr)

def _parallel_correlation(pixel,bkg,arr,coord,smth_time):
    """
    Calculates the Pearson r correlation coefficent between the savgol filtered lightcurve and the upper 30% of the background, at the same indices.

    Parameters:
    ----------
    pixel: ArrayLike
        The flux lightcurve to be filtered and correlated.
    bkg: ArrayLike
        The background lightcurve.

    arr: Not Used But Positional

    coord: Not Used But Positional

    smth_time: int
        The window lenght of the savgol filter, must be <= size of pixel

    Returns:
    -------
    corr: float
        The absolute value of the Pearson r correlation coefficent between the filtered lightcurve and the upper 30% of the background, rounded to 2 decimal places.
    """
    nn = np.isfinite(pixel)
    ff = savgol_filter(pixel[nn],smth_time,2)
    b = bkg[nn]
    indo = (b > np.percentile(b,70)) #& (bb < np.percentile(bb,95))
    corr = pearsonr(ff[indo],b[indo])[0]
    return np.round(abs(corr),2)

def _find_bkg_cor(tess,cores):
    """
    Takes a TESSreduce object and calculates the flux-background Pearson r correlation coefficent in parallel.

    Parameters:
    ----------
    tess: TESSreduce Object
        The TESSreduce object that is needing the correlation coefficents calculated.
    cores: int
        The number of cores to be used for parallel processing.
    
    Returns:
    cors: ArrayLike 
        The array of Pearson r correlation coefficents     

    """
    y,x = np.where(np.isfinite(tess.ref))
    coord = np.c_[y,x]
    cors = np.zeros_like(tess.ref)

    cor = Parallel(n_jobs=cores)(delayed(_parallel_correlation)
                                           (tess.flux[:,coord[i,0],coord[i,1]],
                                            tess.bkg[:,coord[i,0],coord[i,1]],
                                            cors,coord[i],30) for i in range(len(coord)))
    cor = np.array(cor)
    cors[coord[:,0],coord[:,1]] = cor
    return cors

def _address_peaks(flux,bkg,std):  
    """
    Filters the upper 30% of the background values and their corresponding flux values. The fit to the background involves a minimization of the correlation coefficents and an interpolated savgol filter of the fluxes. The fluxes are modified by the same savgol filter, and the median of the lower 16% of the std. 

    Parameters:
    ----------
    flux: ArrayLike
        The flux array of interest
    bkg: ArrayLike
        The background flux array corresponding to flux.
    std: ArrayLike
        An array of the standard deviations of the background
    
    Returns:
    -------
    new_flux: ArrayLike
        The modified flux array. If there is nothing to modify, new_flux==flux.
    new_bkg: ArrayLike
        The modified background array. If there is nothing to modify, new_bkg==bkg.
    """
    
    nn = np.isfinite(flux)
    b = bkg[nn]
    f = flux[nn]
    bkg_ind = (b > np.percentile(b,70)) #& (bb < np.percentile(bb,95))
    split = np.where(np.diff(np.where(bkg_ind)[0]) > 100)[0][0]
    new_bkg = deepcopy(bkg)
    new_flux = deepcopy(flux)
    counter = np.arange(len(b[bkg_ind]),dtype=int)
    for i in range(2):
        if i == 0:
            split_ind = counter[split:]
        else:
            split_ind = counter[:split]
        ff = deepcopy(f[bkg_ind][split_ind])
        s = std[nn][bkg_ind][split_ind]
        med_ind = s < np.percentile(s,16)
        med = np.nanmedian(ff[med_ind])
        ff -= med
        
        x0 =[1e-3]
        fit = minimize(cor_minimizer,x0,(ff,b[bkg_ind][split_ind]),method='Powell')
        ff -= b[bkg_ind][split_ind]*fit.x
        
        
        bound_ind = (ff < s*2) & (ff > -s*2)
        if np.sum(bound_ind*1) > 10:
            xf = np.arange(len(ff))

            sav = savgol_filter(ff[bound_ind],len(ff[bound_ind])//2 + 1,3)
            interp = interp1d(xf[bound_ind],sav,bounds_error=False,fill_value='extrapolate')
            sav = interp(xf)
            
            #plt.figure()
            #plt.plot(new_flux[nn][bkg_ind][split_ind])
            #plt.plot(new_bkg[nn][bkg_ind][split_ind]*fit.x + sav)
            #plt.plot(new_bkg[nn][bkg_ind][split_ind]*fit.x)
            #plt.plot(sav)
            #plt.plot(ff-sav,'--')
        else:
            sav = 0
        indo = np.arange(len(flux))
        indo = indo[nn][bkg_ind][split_ind]
        
        new_bkg[indo] += new_bkg[nn][bkg_ind][split_ind]*fit.x + sav
        new_flux[indo] = ff - sav + med
    return new_flux, new_bkg

def _calc_bkg_std(data,coord,d=6):
    """
    Calculates the background standard deviation of data in a rectangle of size d pixels around the coord point given. 

    Parameters:
    ----------
    data: ArrayLike
        A 2d Array of flux values to calculate the standard deviation of.
    coord: ArrayLike (shape(2,))
        The y, x coordinate to calculate the standard deviation around.
    d: int, optional
        The size of the rectangle to have the standard deviation calculates in. If the pairing of coord and d would result in a rectangle indexing outside of data, this is corrected for, so d is the maximum size of the rectangle, and will give a square box if no corrections are needed. Default is 6.

    Returns:
    ------- 
    std: float
        The standard deviation of data at and around coord.
    """

    y = coord[0]; x = coord[1]
    ylow = y-d; yhigh=y+d+1
    if ylow < 0: 
        ylow=0; 
    if yhigh > data.shape[0]: 
        yhigh=data.shape[0]
    xlow = x-d; xhigh=x+d
    if xlow < 0: 
        xlow=0; 
    if xhigh > data.shape[0]: 
        xhigh=data.shape[0]
        
    std = np.nanstd(data[:,ylow:yhigh,xlow:xhigh],axis=(1,2))
    return std 


def multi_correlation_cor(tess,limit=0.8,cores=7):
    """
    Corrects for correlation coefficents larger than limit. If the flux and the background of tess are correlated (absolute value of correlation coefficent, |r|) to a level higher than limit, a fit to minimize this coefficent is preformed, and the new background and flux values are returned

    Parameters:
    ----------
    tess: TESSreduce Object

    limit: float, optional
        The largest acceptable |r| before any modifications are needed. Should be in range (0,1) for comparison to |r| to make any sense. Default is 0.8. 
    cores: int, optional
        The number of cores to use for multiprocessing. Default is 7.
    
    Returns:
    -------
    flux: ArrayLike
        The modified flux array, after any needed changes have been made. If nothing is needed to be changed, or the modification breaks, flux == tess.flux.
    bkg:
        The modified background array, after any needed changes have been made. If nothing is needed to be changed, or the modification breaks, bkg == tess.bkg.

    """
    cors = _find_bkg_cor(tess,cores=cores)
    y,x = np.where(cors > limit) 
    flux = deepcopy(tess.flux)
    bkg = deepcopy(tess.bkg)
    if len(y > 0):
        try:
            coord = np.c_[y,x]
            dat = tess.bkg
            stds = np.zeros_like(dat)
            std = Parallel(n_jobs=cores)(delayed(_calc_bkg_std)(dat,coord[i])for i in range(len(coord)))
            std = np.array(std)
            stds[:,coord[:,0],coord[:,1]] = std.T
            
            new_flux, new_bkg = zip(*Parallel(n_jobs=cores)(delayed(_address_peaks)
                                                    (tess.flux[:,coord[i,0],coord[i,1]],
                                                     tess.bkg[:,coord[i,0],coord[i,1]],
                                                     stds[:,coord[i,0],coord[i,1]]) 
                                                     for i in range(len(coord))))
            
            new_bkg = np.array(new_bkg)
            new_flux = np.array(new_flux)
            flux[:,coord[:,0],coord[:,1]] = new_flux.T
            bkg[:,coord[:,0],coord[:,1]] = new_bkg.T
        except:
            bad = 1
    return flux, bkg 
    

    
