import pandas as pd
import numpy as np
import requests
import json

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
        Class to gather and organise ground data for transients.
        
        ------
        Inputs
        ------
        ra : float 
            right ascension in decimal degrees 
        dec : float 
            declination in decimal degrees
        sn_name : str 
            transient catalogue name


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
        """
        If coordinates are known then get the transient name from OSC
        """
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
        """
        Cross refeerence the OSC to get selected catalog names 
        
        -----
        Input
        -----
        catalog : str
            shorthand name for catalog, currently only "ztf" is implemented.
        """
        if catalog.lower() != 'ztf':
            m = 'Only ztf is available at this time'
            raise ValueError(m)
        # query the OSC
        url = 'https://api.astrocats.space/{}/alias'.format(self.sn_name)
        response = requests.get(url)
        json_acceptable_string = response.content.decode("utf-8").replace("'", "").split('\n')[0]
        d = json.loads(json_acceptable_string)
        try:
            # check if the lookup failed
            print(d['message'])
            return None
        except:
            pass
        alias = d[self.sn_name]['alias']
        names = [x['value'] for x in alias]
        names = np.array(names)
        ind = [x.lower().startswith(catalog) for x in names]
        #print(names[ind])
        try:
            # return the cotalog specific name if it exists 
            return names[ind][0]
        except:
            return None


    def get_ztf_data(self):
        """
        Gets the ztf light curve data. First checks that the transient name is defined
        """
        if self.sn_name is None:
            self.get_sn_name()
        ztf_name = self.alias(catalog='ztf')
        if ztf_name is not None:
            self.ztf = get_ztf(ztf_name)
        return



    def to_flux(self,flux_type='mjy'):
        """
        Convert the ground based data from magnitude space to selected flux space.

        ------
        Inputs
        ------
        flux_type : str
            flux type to convert to, currently only "mjy", "jy", and "erg"/"cgs" are available.
        """
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

        # Add in other lcs if ever queryable   
        return  