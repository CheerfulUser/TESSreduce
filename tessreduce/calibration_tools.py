import os
dirname = os.path.dirname(__file__)


import astropy.table as at
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import astropy.table as at

import pandas as pd
from glob import glob
from copy import deepcopy
from scipy.optimize import minimize
from astropy.stats import sigma_clip
from .sigmacut import calcaverageclass
from .R_load import R_val

from scipy.interpolate import UnivariateSpline

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

def Save_space(Save):
    """
    Creates a pathm if it doesn't already exist.
    """
    try:
        if not os.path.exists(Save):
            os.makedirs(Save)
    except FileExistsError:
        pass


# Tools to use the Tonry 2012 PS1 color splines to fit extinction

def Tonry_clip(Colours,model):
    """
    Use the Tonry 2012 PS1 splines to sigma clip the observed data.
    """
    #tonry = np.loadtxt(os.path.join(dirname,'Tonry_splines.txt'))
    tonry = model
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs r-i'][0,:]
    mx = tonry[:,0]
    y = Colours['obs g-r'][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = x[:,np.newaxis] - mx[np.newaxis,:]
    yy = y[:,np.newaxis] - my[np.newaxis,:]
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mins = np.nanmin(dd,axis=1)
    # Sigma clip the distance data
    ind = np.isfinite(mins)
    sig = sigma_mask(mins[ind])
    # return sigma clipped mask
    ind[ind] = ~sig
    return ind

def Tonry_residual(Colours,model):
    """
    Calculate the residuals of the observed data from the Tonry et al 2012 PS1 splines.
    """
    tonry = model
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs ' + X][0,:]
    mx = tonry[:,0]
    y = Colours['obs ' + Y][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = (x[:,np.newaxis] - mx[np.newaxis,:]) #.astype(float)
    yy = (y[:,np.newaxis] - my[np.newaxis,:]) #.astype(float)
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mingr = np.nanmin(dd,axis=1)
    return np.nansum(mingr) #+ np.nansum(miniz)

def Tonry_fit(K,Data,Model,Compare,system='ps1'):
    """
    Wrapper for the residuals function
    """
    Colours = Make_colours(Data,Model,Compare,Extinction = K,Redden=False, 
                            Tonry = True, system=system)
    res = Tonry_residual(Colours,Model)
    return res

def Tonry_reduce(Data,plot=False,savename=None,system='ps1'):
    '''
    Uses the Tonry et al. 2012 PS1 splines to fit dust and find all outliers.
    '''
    data = deepcopy(Data)
    if system.lower() == 'ps1':
        tonry = np.loadtxt(os.path.join(dirname,'Tonry_splines.txt'))
    else:
        tonry = np.loadtxt(os.path.join(dirname,'SMspline.txt'))
    compare = np.array([['r-i','g-r']])   
    
    dat = data
    clips = []
    if len(dat) < 10:
        raise ValueError('No data available')
    for i in range(2):
        if i == 0:
            k0 = 0.01
        else:
            k0 = res.x

        res = minimize(Tonry_fit,k0,args=(dat,tonry,compare,system),method='Nelder-Mead')
        
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True,system=system)
        clip = Tonry_clip(colours,tonry)
        clips += [clip]
        dat = dat.iloc[clip]
        #print('Pass ' + str(i+1) + ': '  + str(res.x[0]))
    clips[0][clips[0]] = clips[1]
    if plot:
        orig = Make_colours(dat,tonry,compare,Extinction = 0, Tonry = True,system=system)
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True,system=system)
        plt.figure(figsize=(1.5*fig_width,1*fig_width))
        #plt.title('Fit to Tonry et al. 2012 PS1 stellar locus')
        plt.plot(orig['obs r-i'].flatten(),orig['obs g-r'].flatten(),'C1+',alpha=0.5,label='Raw')
        plt.plot(colours['obs r-i'].flatten(),colours['obs g-r'].flatten(),'C0.',alpha=0.5,label='Corrected')
        plt.plot(colours['mod r-i'].flatten(),colours['mod g-r'].flatten(),'k-',label='Model')
        plt.xlabel('$r-i$',fontsize=15)
        plt.ylabel('$g-r$',fontsize=15)
        plt.text(0.75, 0.25, '$E(B-V)={}$'.format(str(np.round(res.x[0],3))))
        plt.legend()
        plt.show()
        if savename is not None:
            plt.savefig(savename + '_SLR.pdf', bbox_inches = "tight")
    #clipped_data = data.iloc[clips[0]] 
    return res.x, dat



def sigma_mask(data,error= None,sigma=3,Verbose= False):
    if type(error) == type(None):
        error = np.zeros(len(data))
    
    calcaverage = calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    if Verbose:
        print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped



def Get_lcs(X,Y,K,Colours,fitfilt = ''):
    """
    Make the colour combinations
    """
    keys = np.array(list(Colours.keys()))

    xind = 'mod ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'mod ' + Y == keys
    y = Colours[keys[yind][0]]

    #x_interp = np.arange(np.nanmin(x),0.8,0.01)
    #inter = interpolate.interp1d(x,y)
    #l_interp = inter(x_interp)
    locus = np.array([x,y])

    xind = 'obs ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'obs ' + Y == keys
    y = Colours[keys[yind][0]]
    c1,c2 = X.split('-')
    c3,c4 = Y.split('-')
    # parameters
    ob_x = x.copy() 
    ob_y = y.copy() 

    if c1 == fitfilt: ob_x[0,:] += K
    if c2 == fitfilt: ob_x[0,:] -= K

    if c3 == fitfilt: ob_y[0,:] += K
    if c4 == fitfilt: ob_y[0,:] -= K
    return ob_x, ob_y, locus

def Dot_prod_error(x,y,Model):
    """
    Calculate the error projection in the direction of a selected point.
    ------------------
    Currently not used
    ------------------
    """
    #print(Model.shape)
    adj = y[0,:] - Model[1,:]
    op = x[0,:] - Model[0,:]
    #print(adj.shape,op.shape)
    hyp = np.sqrt(adj**2 + op**2)
    costheta = adj / hyp
    yerr_proj = abs(y[1,:] * costheta)
    xerr_proj = abs(x[1,:] * costheta)
    
    proj_err = yerr_proj + xerr_proj
    #print(proj_err)
    return proj_err 


def Dist_tensor(X,Y,K,Colours,fitfilt='',Tensor=False,Plot = False):
    """
    Calculate the distance of sources in colour space from the model stellar locus.
    
    ------
    Inputs
    ------
    X : str
        string containing the colour combination for the X axis 
    Y : str
        string containing the colour combination for the Y axis 
    K : str 
        Not sure...
    Colours : dict
        dictionary of colour combinations for all sources 
    fitfilt : str 
         Not used...
     Tensor : bool
        if true this returns the distance tensor instead of the total sum
    Plot : bool
        if true this makes diagnotic plots

    -------
    Returns
    -------
    residual : float
        residuals of distances from all points to the model locus. 
    """
    ob_x, ob_y, locus = Get_lcs(X,Y,K,Colours,fitfilt)
    
    ind = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))[0]
    indo = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))
    ob_x = ob_x[:,ind]
    ob_y = ob_y[:,ind]
    
    
    if Plot:
        plt.figure()
        plt.title(X + ' ' + Y)
        plt.plot(ob_x[0,:],ob_y[0,:],'.')
        plt.plot(locus[0,:],locus[1,:])
    #print(ob_x.shape)
    #print('x ',ob_x.shape[1])

    x = np.zeros((ob_x.shape[1],locus.shape[1])) + ob_x[0,:,np.newaxis]
    x -= locus[0,np.newaxis,:]
    y = np.zeros((ob_y.shape[1],locus.shape[1])) + ob_y[0,:,np.newaxis]
    y -= locus[1,np.newaxis,:]

    dist_tensor = np.sqrt(x**2 + y**2)
    #print(np.nanmin(dist_tensor,axis=1))
    #print(X + Y +' dist ',dist_tensor.shape)
    
    if len(dist_tensor[np.isfinite(dist_tensor)]) > 1:
        minind = np.nanargmin(abs(dist_tensor),axis=1)
        mindist = np.nanmin(abs(dist_tensor),axis=1)
        sign = (ob_y[0,:] - locus[1,minind])
        sign = sign / abs(sign)

        eh = mindist * sign
    
        proj_err = Dot_prod_error(ob_x,ob_y,locus[:,minind])
        #print('mindist ',mindist)
        if Tensor:
            return eh
        if len(mindist) > 0:
            #print('thingo',np.nanstd(mindist))
            residual = np.nansum(abs(mindist)) #/ proj_err)
        else:
            #print('infs')
            residual = np.inf
    else:
        if Tensor:
            return []
        residual = np.inf
        #residual += 100*np.sum(np.isnan(dist))
    #print(residual)
    cut_points = len(indo) - len(ind)
    return residual + cut_points * 100


def Make_colours(Data, Model, Compare, Extinction = 0, Redden = False,Tonry=False,system='ps1'):
    #R = {'g': 3.518, 'r':2.617, 'i':1.971, 'z':1.549, 'y': 1.286, 'k':2.431,'tess':1.809}#'z':1.549} # value from bayestar
    R = {'g': 3.61562687, 'r':2.58602003, 'i':1.90959054, 'z':1.50168735, 
         'y': 1.25340149, 'kep':2.68629375,'tess':1.809}
    gr = (Data['gmag'] - Data['rmag']).values
    colours = {}
    for x,y in Compare:
        colours['obs ' + x] = np.array([Data[x.split('-')[0]+'mag'].values - Data[x.split('-')[1]+'mag'].values,
                                        Data['e_'+x.split('-')[0]+'mag'].values - Data['e_'+x.split('-')[1]+'mag'].values])
        colours['obs ' + y] = np.array([Data[y.split('-')[0]+'mag'].values - Data[y.split('-')[1]+'mag'].values,
                                        Data['e_'+y.split('-')[0]+'mag'].values - Data['e_'+y.split('-')[1]+'mag'].values])
        if Tonry:
            colours['mod ' + x] = Model[:,0]
            colours['mod ' + y] = Model[:,1]
        else:

            xx = Model[x.split('-')[0]] - Model[x.split('-')[1]]
            yy = Model[y.split('-')[0]] - Model[y.split('-')[1]]
            ind = xx.argsort()
            xx = xx[ind]
            yy = yy[ind]
            spl = UnivariateSpline(xx, yy)
            c_range = np.arange(xx[0],0.8,0.01)
            colours['mod ' + x] = c_range
            colours['mod ' + y] = spl(c_range)
        
        if Redden:
            colours['mod ' + x] += Extinction*(R_val(x.split('-')[0],gr=gr,system=system)[0] - R_val(x.split('-')[1],gr=gr,system=system)[0])
            colours['mod ' + y] += Extinction*(R_val(y.split('-')[0],gr=gr,system=system)[0] - R_val(y.split('-')[1],gr=gr,system=system)[0])
        else:
            colours['obs ' + x] -= Extinction*(R_val(x.split('-')[0],gr=gr,system=system)[0] - R_val(x.split('-')[1],gr=gr,system=system)[0])
            colours['obs ' + y] -= Extinction*(R_val(y.split('-')[0],gr=gr,system=system)[0] - R_val(y.split('-')[1],gr=gr,system=system)[0])
    return colours 


def Isolated_stars(pos,Tmag,flux,Median, Distance = 7, Aperture=3, Mag = 16):
    """
    Find isolated stars in the scene.
    """
    
    #pos, Tmag = sd.Get_PS1(tpf,magnitude_limit=18)
    pos_shift = pos + 0.5
    ind = ((Distance//2< pos_shift[:,0]) & (pos_shift[:,0]< flux.shape[1]-Distance//2) & 
          (Distance//2< pos_shift[:,1]) & (pos_shift[:,1]< flux.shape[1]-Distance//2) &
          (Tmag < Mag))
    
    if ~ind.any():
        raise ValueError('No sources brighter than {} Tmag.'.format(Mag))
    p = pos_shift[ind,:]
    
    distance= np.zeros([len(p),len(p)])
    for i in range(len(p)):
        distance[i] = np.sqrt((p[i,0] - p[:,0])**2 + (p[i,1] - p[:,1])**2)
    distance[distance==0] = np.nan
    mins = np.nanmin(distance,axis=1)
    
    iso = p[mins > Distance]
    iso = iso.astype('int')
    ind[ind] = mins > Distance
    median = Median
    median[median<0] = 0
    if len(iso)> 0:
        clips = []
        time_series = []
        Distance = Aperture
        if (Distance % 2) ==0:
            d = Distance - 1
        else:
            d = Distance
        u = d//2 +1
        l = d //2 

        for i in range(len(iso)):
            clips += [median[iso[i,1]-l:iso[i,1]+u,iso[i,0]-l:iso[i,0]+u]]
            time_series += [flux[:,iso[i,1]-l:iso[i,1]+u,iso[i,0]-l:iso[i,0]+u]]
        #print(clips)
        clips=np.array(clips)
        time_series=np.array(time_series)
    else:
        raise ValueError('No stars brighter than {} Tmag and isolated by {} pix. Concider lowering brightness.'.format(Mag,Distance))
    return ind, clips, time_series