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
    # sd = sd.append(sn)
    sd = pd.concat([sd, sn], ignore_index=True)
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


    def __old_get_sn_name(self):
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

    def get_sn_name(self):
        ra = self.ra
        dec = self.dec
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
            url = f'https://www.wis-tns.org/search?&reported_within_last_value=&reported_within_last_units=days&unclassified_at=0&classified_sne=0&classified_tde=0&include_frb=1&name=&name_like=0&isTNS_AT=all&public=all&unreal=no&ra={ra}&decl={dec}&radius=10&coords_unit=arcsec&reporting_groupid%5B%5D=&groupid%5B%5D=&classifier_groupid%5B%5D=&objtype%5B%5D=&at_type%5B%5D=&discovery_date_start=&discovery_date_end=&discovery_mag_min=&discovery_mag_max=&internal_name=&discoverer=&classifier=&spectra_count=&redshift_min=&redshift_max=&hostname=&ext_catid=&ra_range_min=&ra_range_max=&decl_range_min=&decl_range_max=&discovery_instrument%5B%5D=&classification_instrument%5B%5D=&associated_groups%5B%5D=&official_discovery=0&official_classification=0&auto_classification_algorithm%5B%5D=&auto_classification_objtypeid%5B%5D=&auto_classification_prob=&at_rep_remarks=&class_rep_remarks=&frb_repeat=&frb_repeater_of_objid=&frb_measured_redshift=0&frb_dm_range_min=&frb_dm_range_max=&frb_rm_range_min=&frb_rm_range_max=&frb_snr_range_min=&frb_snr_range_max=&frb_flux_range_min=&frb_flux_range_max=&num_page=50&display%5Bredshift%5D=1&display%5Bhostname%5D=1&display%5Bhost_redshift%5D=1&display%5Bsource_group_name%5D=1&display%5Bclassifying_source_group_name%5D=1&display%5Bdiscovering_instrument_name%5D=0&display%5Bclassifing_instrument_name%5D=0&display%5Bprograms_name%5D=0&display%5Binternal_name%5D=1&display%5BisTNS_AT%5D=0&display%5Bpublic%5D=1&display%5Bend_pop_period%5D=0&display%5Bspectra_count%5D=1&display%5Bdiscoverymag%5D=1&display%5Bdiscmagfilter%5D=1&display%5Bdiscoverydate%5D=1&display%5Bdiscoverer%5D=1&display%5Bremarks%5D=0&display%5Bsources%5D=0&display%5Bbibcode%5D=0&display%5Bext_catalogs%5D=0&display%5Bunreal%5D=0&display%5Brepeater_of_objid%5D=0&display%5Bdm%5D=0&display%5Bgalactic_max_dm%5D=0&display%5Bbarycentric_event_time%5D=0&display%5Bpublic_webpage%5D=0'
            search = requests.get(url, headers=headers)

            name = search.text.split('<td class="cell-name"><a href="https://www.wis-tns.org/object/')[1].split('>')[1].split('<')[0].split(' ')[1]
            self.sn_name = name
            url = f'https://www.wis-tns.org/object/{name}' # hard coding in that the event is in the 2000s
            
            result = requests.get(url, headers=headers)

            n = result.text.split('<td class="cell-internal_name">')[1:]
            names = []
            for i in range(len(n)):
                names 
                if '</td>\n                      <td class="cell-groups">' in n[i]:
                    names += [n[i].split('</td>')[0]]
                    
            names = list(set(names))
            self.cat_names = names
        except:
            print('No reported transients within 10arcsec of coordinates')
            self.sn_name = None
            self.cat_names = None


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
        for name in self.cat_names:
            if 'ZTF' in name:
                ztf_name = name
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