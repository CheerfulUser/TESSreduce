import pandas as pd
import numpy as np
import requests
a_current = True
try:
    from alerce.core import Alerce
    client = Alerce()
except:
    from alerce.api import AlerceAPI  # old API
    client = AlerceAPI()
    print('WARNING: using old Alerce API')
    a_current = False
import json


def get_ztf(oid):
    # query detections
    if a_current:
        try:
            sd = client.query_detections(oid, format='pandas')
        except:
            sd = client.query_detections(oid, format='pandas')
    else:  # old Alerce API
        try:
            sd = client.get_detections(oid, format='pandas')
        except:
            sd = client.query_detections(oid, format='pandas')
    sd = sd.sort_values("mjd")
        
    # query non detections
    try:
        sn = client.query_non_detections(oid, format='pandas')
    except:  # old Alerce API
        sn = client.get_non_detections(oid, format='pandas')
        if sn.index.name == 'mjd':
            sn.reset_index(level=0, inplace=True)
            
    sn = sn.sort_values("mjd")

    sd = sd[['mjd','magpsf','sigmapsf','fid']]
    sd['maglim'] = np.nan#SN_det['diffmaglim']
    sn = sn[['mjd','diffmaglim','fid']]
    sn = sn.rename(columns={'diffmaglim':'maglim'})
    sn['magpsf'] = np.nan
    sn['sigmapsf'] = np.nan
    sd = sd.append(sn)
    sd = sd.sort_values('mjd',ignore_index=True)
    sd = sd.rename(columns={'magpsf':'mag','sigmapsf':'mag_e'})
    sd['flux'] = np.nan
    sd['flux_e'] = np.nan
    sd['fluxlim'] = np.nan
    sd.fid.iloc[sd.fid.values==1] = 'g'
    sd.fid.iloc[sd.fid.values==2] = 'r'
    return sd


class ground():
    def __init__(self,ra=None,dec=None,sn_name=None):
        """
        Class to reduce tess data.
        """
        self.ra  = ra
        self.dec  = dec 
        self.sn_name  = sn_name

        # diags
        self.zp = -48.6
        self.flux_type = 'mag'


        #calculate
        self.ztf = None
        self.asassn = None
        self.ps1 = None


    def get_sn_name(self):
        url = 'https://api.astrocats.space/catalog?ra={ra}&dec={dec}&closest'.format(ra = self.ra, dec = self.dec)
        response = requests.get(url)
        json_acceptable_string = response.content.decode("utf-8").replace("'", "").split('\n')[0]
        d = json.loads(json_acceptable_string)
        try:
            print(d['message'])
            self.sn_name = None
            return 
        except:
            self.sn_name = list(d.keys())[0]
            return

    def alias(self,catalog='ztf'):
        url = 'https://api.astrocats.space/{}/alias'.format(self.sn_name)
        response = requests.get(url)
        json_acceptable_string = response.content.decode("utf-8").replace("'", "").split('\n')[0]
        d = json.loads(json_acceptable_string)
        try:
            print(d['message'])
            return None
        except:
            pass
        alias = d[self.sn_name]['alias']
        names = [x['value'] for x in alias]
        names = np.array(names)
        ind = [x.lower().startswith(catalog) for x in names]
        print(names[ind])
        try:
            return names[ind][0]
        except:
            return None


    def get_ztf(self):
        if self.sn_name is None:
            self.get_sn_name()
        ztf_name = self.alias(catalog='ztf')
        if ztf_name is not None:
            self.ztf = get_ztf(ztf_name)
        return



    def to_flux(self,flux_type='mjy'):

        if flux_type.lower() == 'mjy':
            flux_zp = 16.4
            self.flux_type = 'mJy'
        elif flux_type.lower() == 'jy':
            flux_zp = 8.9
            self.flux_type = 'Jy'
        elif (flux_type.lower() == 'erg') | (flux_type.lower() == 'cgs'):
            flux_zp = -48.6
            self.flux_type = 'cgs'
        else:
            m = '"'+flux_type + '" is not a valid option, please choose from:\njy\nmjy\ncgs/erg'
            raise ValueError(m)

        self.ztf['flux'] = 10**((self.ztf['mag'].values - flux_zp)/-2.5)
        self.ztf['flux_e'] = self.ztf['flux'].values * self.ztf['mag_e'].values * np.log(10)/2.5
        self.ztf['fluxlim'] = 10**((self.ztf['maglim'].values - flux_zp)/-2.5)

        # Add in other lcs if ever get 
        return  


def plot_ztf_target(oid,fig=None):
    """ Given a ZTF target id, this will plot a ZTF light curve from Alerce.
    
    Parameters
    ----------
    oid : str
        ZTF target id.
        
    """
        
    SN_det, SN_nondet = get_ztf(oid)

    # plotting properties
    labels = {1: 'g', 2: 'r'}
    markers = {1: '--o', 2: '--s'}
    sizes = {1: 60, 2: 60}
    colors = {1: '#56E03A', 2: '#D42F4B'}  # color blind friendly green and red 
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 7))
      
    # loop the passbands
    for fid in ['g', 'r']:
        
        # plot detections if available
        mask = SN_det.fid == fid
        if np.sum(mask) > 0:
            # note that the detections index is candid and that we are plotting the psf corrected magnitudes
            plt.errorbar(SN_det[mask].mjd, SN_det[mask].magpsf, 
                yerr = SN_det[mask].sigmapsf, c=colors[fid], label=labels[fid], fmt=markers[fid])
        
        # plot non detections if available
        mask = (SN_nondet.fid == fid) & (SN_nondet.diffmaglim > -900)
        if np.sum(mask) > 0:     
            # non detections index is mjd
            plt.scatter(SN_nondet[mask].mjd, SN_nondet[mask].diffmaglim, c=colors[fid], alpha = 0.6,
                marker='v', s=sizes[fid])
    if fig is None:    
        ax.set_xlabel('MJD', fontsize=14)
        ax.set_ylabel('Apparent Mag.', fontsize=14)
        ax.set_title(oid, fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().invert_yaxis()
        ax.legend(frameon=False,fontsize=16)
    return