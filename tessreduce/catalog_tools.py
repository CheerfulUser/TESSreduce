import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from scipy import interpolate
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from copy import deepcopy
import pandas as pd
from .R_load import R_val

def Get_Catalogue(tpf, Catalog = 'gaia'):
	"""
	Get the coordinates and mag of all sources in the field of view from a specified catalogue.


	I/347/gaia2dis   Distances to 1.33 billion stars in Gaia DR2 (Bailer-Jones+, 2018)

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		Catalogue 			str 	Permitted options: 'gaia', 'dist', 'ps1'
	
	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
	# Use pixel scale for query size
	pix_scale = 4.0  # arcseconds / pixel for Kepler, default
	if tpf.mission == 'TESS':
		pix_scale = 21.0
	# We are querying with a diameter as the radius, overfilling by 2x.
	from astroquery.vizier import Vizier
	Vizier.ROW_LIMIT = -1
	if Catalog == 'gaia':
		catalog = "I/345/gaia2"
	elif Catalog == 'dist':
		catalog = "I/350/gaiaedr3"
	elif Catalog == 'ps1':
		catalog = "II/349/ps1"
	elif Catalog == 'skymapper':
		catalog = 'II/358/smss'
	else:
		raise ValueError("{} not recognised as a catalog. Available options: 'gaia', 'dist','ps1'")
	if Catalog == 'gaia':
		result = Vizier.query_region(c1, catalog=[catalog],
                             		 radius=Angle(1890, "arcsec"),column_filters={'Gmag':'<19'})
	else:
		result = Vizier.query_region(c1, catalog=[catalog],
									 radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))

	no_targets_found_message = ValueError('Either no sources were found in the query region '
										  'or Vizier is unavailable')
	#too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
	if result is None:
		raise no_targets_found_message
	elif len(result) == 0:
		raise no_targets_found_message
	result = result[catalog].to_pandas()
	
	return result 


def Get_Gaia(tpf, magnitude_limit = 18, Offset = 10):
	"""
	Get the coordinates and mag of all gaia sources in the field of view.

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		magnitude_limit 	float 	cutoff for Gaia sources
		Offset 				int 	offset for the boundary 
	
	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	keys = ['objID','RAJ2000','DEJ2000','e_RAJ2000','e_DEJ2000','gmag','e_gmag','gKmag','e_gKmag','rmag',
			'e_rmag','rKmag','e_rKmag','imag','e_imag','iKmag','e_iKmag','zmag','e_zmag','zKmag','e_zKmag',
			'ymag','e_ymag','yKmag','e_yKmag','tmag','gaiaid','gaiamag','gaiadist','gaiadist_u','gaiadist_l',
			'row','col']

	result =  Get_Catalogue(tpf, Catalog = 'gaia')

	result = result[result.Gmag < magnitude_limit]
	if len(result) == 0:
		raise no_targets_found_message
	radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
	coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?
	Gmag = result['Gmag'].values
	#Jmag = result['Jmag']
	ind = (((coords[:,0] >= -10) & (coords[:,1] >= -10)) & 
		   ((coords[:,0] < (tpf.shape[1] + 10)) & (coords[:,1] < (tpf.shape[2] + 10))))
	coords = coords[ind]
	Gmag = Gmag[ind]
	Tmag = Gmag - 0.5
	#Jmag = Jmag[ind]
	return coords, Tmag


def mag2flux(mag,zp):
    f = 10**(2/5*(zp-mag))
    return f



def PS1_to_TESS_mag(PS1,ebv = 0):
    zp = 25
    gr = (PS1.gmag - PS1.rmag).values
    
    eg, e = R_val('g',gr=gr,ext=ebv); er, e = R_val('r',gr=gr,ext=ebv)
    ei, e = R_val('i',gr=gr,ext=ebv); ez, e = R_val('z',gr=gr,ext=ebv)
    ey, e = R_val('y',gr=gr,ext=ebv); et, e = R_val('tess',gr=gr,ext=ebv)
    eg = eg  * ebv; er = er  * ebv; ei = ei  * ebv; ez = ez  * ebv
    ey = ey  * ebv; et = et  * ebv

    g = mag2flux(PS1.gmag.values - eg,zp)
    r = mag2flux(PS1.rmag.values - er,zp)
    i = mag2flux(PS1.imag.values - ei,zp)
    z = mag2flux(PS1.zmag.values - ez,zp)
    y = mag2flux(PS1.ymag.values - ey,zp)
    
    cr = 0.25582823; ci = 0.27609407; cz = 0.35809516
    cy = 0.11244277; cp = 0.00049096

    t = (cr*r + ci*i + cz*z + cy*y)*(g/i)**cp
    t = -2.5*np.log10(t) + zp + et
    PS1['tmag'] = t
    return PS1

def SM_to_TESS_mag(SM,ebv = 0):
    zp = 25
    gr = (SM.gmag - SM.rmag).values
    
    eg, e = R_val('g',gr=gr,ext=ebv,system='skymapper')
    er, e = R_val('r',gr=gr,ext=ebv,system='skymapper')
    ei, e = R_val('i',gr=gr,ext=ebv,system='skymapper') 
    ez, e = R_val('z',gr=gr,ext=ebv,system='skymapper')
    et, e = R_val('tess',gr=gr,ext=ebv)
    eg = eg  * ebv; er = er  * ebv; ei = ei  * ebv
    ez = ez  * ebv; et = et  * ebv

    g = mag2flux(SM.gmag.values - eg,zp)
    r = mag2flux(SM.rmag.values - er,zp)
    i = mag2flux(SM.imag.values - ei,zp)
    z = mag2flux(SM.zmag.values - ez,zp)
    
    cr = 0.25825435; ci = 0.35298213
    cz = 0.39388206; cp = -0.00170817

    t = (cr*r + ci*i + cz*z)*(g/i)**cp
    t = -2.5*np.log10(t) + zp + et
    SM['tmag'] = t
    return SM



def Get_PS1(tpf, magnitude_limit = 18, Offset = 10):
	"""
	Get the coordinates and mag of all PS1 sources in the field of view.

	-------
	Inputs-
	-------
		tpf 				class 	target pixel file lightkurve class
		magnitude_limit 	float 	cutoff for Gaia sources
		Offset 				int 	offset for the boundary 

	--------
	Outputs-
	--------
		coords 	array	coordinates of sources
		Gmag 	array 	Gmags of sources
	"""
	result =  Get_Catalogue(tpf, Catalog = 'ps1')
	result = result[np.isfinite(result.rmag) & np.isfinite(result.imag)]# & np.isfinite(result.zmag)& np.isfinite(result.ymag)]
	result = PS1_to_TESS_mag(result)
	
	
	result = result[result.tmag < magnitude_limit]
	if len(result) == 0:
		raise no_targets_found_message
	radecs = np.vstack([result['RAJ2000'], result['DEJ2000']]).T
	coords = tpf.wcs.all_world2pix(radecs, 0) ## TODO, is origin supposed to be zero or one?
	Tessmag = result['tmag'].values
	#Jmag = result['Jmag']
	ind = (((coords[:,0] >= -10) & (coords[:,1] >= -10)) & 
		   ((coords[:,0] < (tpf.shape[1] + 10)) & (coords[:,1] < (tpf.shape[2] + 10))))
	coords = coords[ind]
	Tessmag = Tessmag[ind]
	#Jmag = Jmag[ind]
	return coords, Tessmag


def Skymapper_df(sm):
    a = np.zeros(len(sm['ObjectId']),dtype=object)
    a[:] = 's'
    b = sm['ObjectId'].values.astype(str).astype(object)
    obj = a+b
    
    keep = ['objID','RAJ2000', 'DEJ2000','e_RAJ2000','e_DEJ2000','gmag', 'e_gmag', 'gKmag',
           'e_gKmag', 'rmag', 'e_rmag', 'rKmag', 'e_rKmag',
           'imag', 'e_imag', 'iKmag', 'e_iKmag', 'zmag', 'e_zmag',
           'zKmag', 'e_zKmag', 'ymag', 'e_ymag', 'yKmag', 'e_yKmag',
           'tmag']
    df = pd.DataFrame(columns=keep)
    df['objID'] = obj
    df['RAJ2000'] = sm['RAICRS'].values
    df['DEJ2000'] = sm['DEICRS'].values

    df['e_RAJ2000'] = sm['e_RAICRS'].values
    df['e_DEJ2000'] = sm['e_DEICRS'].values

    df['gmag'] = sm['gPSF'].values
    df['rmag'] = sm['rPSF'].values
    df['imag'] = sm['iPSF'].values
    df['zmag'] = sm['zPSF'].values

    df['e_gmag'] = sm['gPSF'].values * np.nan
    df['e_rmag'] = sm['rPSF'].values * np.nan
    df['e_imag'] = sm['iPSF'].values * np.nan
    df['e_zmag'] = sm['zPSF'].values * np.nan

    df['gKmag'] = sm['gPetro'].values
    df['rKmag'] = sm['rPetro'].values
    df['iKmag'] = sm['iPetro'].values
    df['zKmag'] = sm['zPetro'].values
    return df


def Unified_catalog(tpf,magnitude_limit=18,offset=10):
	"""
	Find all sources present in the TESS field from PS!, and Gaia. Catalogs are cross
	matched through distance, and Gaia distances are assigned from Gaia ID.
	Returns a pandas dataframe with all relevant catalog information
	
	------
	Input-
	------
		tpf  lk.Targetpixelfile  target pixel file of the TESS region
		
	-------
	Output-
	-------
		result pd.DataFrame	 Combined catalog
	"""
	
	pd.options.mode.chained_assignment = None
	# need to look at how the icrs coords are offset from J2000
	# Get gaia catalogs 
	gaia = Get_Catalogue(tpf, Catalog = 'gaia')
	gaiadist = Get_Catalogue(tpf, Catalog = 'dist')
	# Get PS1 and structure it
	if tpf.dec > -30:
		ps1 = Get_Catalogue(tpf, Catalog = 'ps1')
		ps1 = ps1[np.isfinite(ps1.rmag) & np.isfinite(ps1.imag)]# & np.isfinite(result.zmag)& np.isfinite(result.ymag)]
		ps1 = PS1_to_TESS_mag(ps1)
		keep = ['objID','RAJ2000', 'DEJ2000','e_RAJ2000','e_DEJ2000','gmag', 'e_gmag', 'gKmag',
			   'e_gKmag', 'rmag', 'e_rmag', 'rKmag', 'e_rKmag',
			   'imag', 'e_imag', 'iKmag', 'e_iKmag', 'zmag', 'e_zmag',
			   'zKmag', 'e_zKmag', 'ymag', 'e_ymag', 'yKmag', 'e_yKmag',
			   'tmag']
		result = ps1[keep]
	else:
		sm = Get_Catalogue(tpf, Catalog = 'skymapper')
		sm = Skymapper_df(sm)
		sm = sm[np.isfinite(sm.rmag) & np.isfinite(sm.imag)]# & np.isfinite(result.zmag)& np.isfinite(result.ymag)]
		sm = PS1_to_TESS_mag(sm)
		result = sm
	# Define the columns for Gaia information
	result['gaiaid'] = 0
	result['gaiaid'] = result['gaiaid'].astype(int)
	result['gaiamag'] = np.nan
	result['gaiadist'] = np.nan
	result['gaiadist_u'] = np.nan
	result['gaiadist_l'] = np.nan
	# Set up arrays to calculate the distance between all PS1 and Gaia sources
	dra = np.zeros((len(gaia),len(result)))
	dra = dra + gaia.RA_ICRS.values[:,np.newaxis]
	dra = dra - result.RAJ2000.values[np.newaxis,:]

	dde = np.zeros((len(gaia),len(result)))
	dde = dde + gaia.DE_ICRS.values[:,np.newaxis]
	dde = dde - result.DEJ2000.values[np.newaxis,:]
	# Calculate distance
	dist = np.sqrt(dde**2 + dra**2)
	
	if isinstance(dist,np.ndarray):
		ind = np.argmin(dist,axis=1)

		far = dist <= (1/60**2) * 1 # difference smaller than 1 arcsec
		# Get index of all valid matches and add the Gaia info
		indo = np.nansum(far,axis=1) > 0
		ind = ind[indo]
		result.gaiaid.iloc[ind] = gaia.Source.values[indo]
		result.gaiamag.iloc[ind] = gaia.Gmag.values[indo]
		#result.tmag.iloc[ind] = gaia.Gmag.values[indo] - .5
		# Add Gaia sources without matches to the dataframe
		keys = list(result.keys())
		indo = np.where(~indo)[0]
		for i in indo:
			df = pd.DataFrame(columns=keys)
			row = np.zeros(len(keys)) * np.nan
			df.RAJ2000 = [gaia.RA_ICRS[i]]; df.DEJ2000 = [gaia.DE_ICRS[i]] 
			df.gaiaid = [gaia.Source[i]]; df.gaiamag = [gaia.Gmag[i]]
			#df.tmag = [gaia.Gmag[i] - 0.5] 
			result = result.append(df,ignore_index=True)

		# Find matches from the distance catalog and add them in
		if False:
			s = np.zeros((len(gaiadist),len(result)))
			s = s + gaiadist.Source.values[:,np.newaxis]
			s = s - result.gaiaid.values[np.newaxis,:]
			inds = np.where(s == 0)[0]
			ind = np.zeros(len(result))
			ind[inds] = 1
			ind = ind > 0
			print(max(ind),len(result))
			print('ind ',ind,type(ind))
			result.gaiadist.iloc[ind] = gaiadist.rest
			result.gaiadist_u.iloc[ind] = gaiadist.B_rest
			result.gaiadist_l.iloc[ind] = gaiadist.b_rest
	
	result = result.iloc[result.tmag.values < magnitude_limit]
	no_targets_found_message = ('Either no sources were found in the query region '
										  'or Vizier is unavailable')
	if len(result) == 0:
		print(no_targets_found_message)

	radecs = np.vstack([result['RAJ2000'], result['DEJ2000']]).T
	coords = tpf.wcs.all_world2pix(radecs, 0)
	result['row'] = coords[:,1]
	result['col'] = coords[:,0]
	#Jmag = result['Jmag']
	ind = (((coords[:,0] >= -offset) & (coords[:,1] >= -offset)) & 
		   ((coords[:,0] < (tpf.shape[1] + offset)) & (coords[:,1] < (tpf.shape[2] + offset))))
	result = result.iloc[ind]
	
	return result

def Reformat_df(df):
    new_cols = ['objID', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000', 'gMeanPSFMag',
               'gMeanPSFMagErr', 'gKmag', 'e_gKmag', 'rMeanPSFMag', 'rMeanPSFMagErr', 'rKmag', 'e_rKmag',
               'iMeanPSFMag', 'iMeanPSFMagErr', 'iKmag', 'e_iKmag', 'zMeanPSFMag', 'zMeanPSFMagErr', 'zKmag',
               'e_zKmag', 'yMeanPSFMag', 'yMeanPSFMagErr', 'yKmag', 'e_yKmag', 'tmag', 'gaiaid',
               'gaiamag', 'gaiadist', 'gaiadist_u', 'gaiadist_l', 'row', 'col']
    new_df = deepcopy(df)
    new_df.columns = new_cols
    
    return new_df