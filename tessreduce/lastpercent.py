import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from copy import deepcopy

def cor_minimizer(coeff,pix_lc,bkg_lc):
    """
    
    """
    lc = pix_lc - coeff * bkg_lc
    ind = np.isfinite(lc) & np.isfinite(bkg_lc)
    #bkgnorm = bkg_lc/np.nanmax(bkg_lc)
    #pixnorm= (lc - np.nanmedian(lc))
    #pixnorm = pixnorm / np.nanmax(abs(pixnorm))
    corr = pearsonr(lc[ind],bkg_lc[ind])[0]
    return abs(corr)

def _parallel_correlation(pixel,bkg,arr,coord,smth_time):
    nn = np.isfinite(pixel)
    ff = savgol_filter(pixel[nn],smth_time,2)
    b = bkg[nn]
    indo = (b > np.percentile(b,70)) #& (bb < np.percentile(bb,95))
    corr = pearsonr(ff[indo],b[indo])[0]
    return np.round(abs(corr),2)

def _find_bkg_cor(tess,cores):
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
    

    
