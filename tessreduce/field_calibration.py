"""
Scene-modelling photometric calibration.

Two additive calibration pathways, both built on the shared scene-fit engine
in `scene_photom.py` (real pixel-space deblending, formal error propagation),
sitting alongside (not replacing) `tessreduce.py`'s existing field_calibrate():

    calibrate_ps1_skymapper(tess, ...)  -- PS1/SkyMapper, with the existing
        Tonry-locus extinction correction and multi-band synthetic-TESS-mag
        reconstruction (both already AB-consistent).
    calibrate_gaia(tess, ...)           -- Gaia DR3 (queried via astroquery,
        matching tessreduce's existing catalog-fetch convention), calibrated
        directly against Rp with the standard Vega->AB offset applied. No
        extinction step, matching the simpler single-band reference.

On any failure (too few good stars, fits don't converge) a warning is raised
and a fixed zp_ab=20.6 fallback is used, rather than raising an exception.
"""
import os
import warnings
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, Angle

from . import scene_photom as sp
from .catalog_tools import Get_Catalogue, PS1_to_TESS_mag, SM_to_TESS_mag
from .calibration_tools import Tonry_reduce

# Gaia DR3 Rp Vega -> AB offset, from synthetic photometry (pysynphot Vega
# spectrum through the actual Gaia3 Rp passband, via calibrimbore's
# get_pb_zpt): zp_AB - zp_Vega = 0.379. The previously-used literature value
# of 0.152 (Casagrande & VandenBerg 2018) was ~0.23 mag too small and was the
# dominant cause of the zp offset between this pathway and PS1/SkyMapper.
GAIA_RP_AB_OFFSET = 0.379

_ZP_FALLBACK = 20.6
_ZP_FALLBACK_ERR = 0.1


def _zp_to_scale(zp):
    """AB zeropoint (mag) -> linear flux-scale factor."""
    return 10 ** (0.4 * np.asarray(zp, dtype=float))


def _scale_to_zp(f):
    """Linear flux-scale factor -> AB zeropoint (mag)."""
    return 2.5 * np.log10(f)


def _zp_fallback(reason):
    warnings.warn(f'field_calibration: {reason} -- falling back to zp_ab={_ZP_FALLBACK}')
    return _ZP_FALLBACK, _ZP_FALLBACK_ERR


def _binned_zeropoint(mag, zp, bin_width=0.5, min_bin_n=5):
    """Magnitude-binned, log-space-safe robust zeropoint combine.

    Bins calibration stars by magnitude, takes a robust (median/1.4826*MAD)
    zeropoint per bin in LINEAR flux-scale space (avoiding the bias of
    averaging a log quantity directly), then inverse-variance-weights the
    per-bin values back into a single zeropoint. Falls back to a global
    robust combine if there are too few usable bins.
    """
    mag = np.asarray(mag, dtype=float)
    zp = np.asarray(zp, dtype=float)
    finite = np.isfinite(mag) & np.isfinite(zp)
    mag, zp = mag[finite], zp[finite]

    def _global():
        scale = _zp_to_scale(zp)
        med = np.median(scale)
        mad = 1.4826 * np.median(np.abs(scale - med))
        return _scale_to_zp(med), (mad / med) * 1.0857, None

    if len(zp) < min_bin_n:
        return _global()

    lo, hi = np.floor(mag.min() / bin_width) * bin_width, np.ceil(mag.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 3:
        return _global()

    bin_scale, bin_mad, bin_n, bin_centers = [], [], [], []
    for i in range(len(edges) - 1):
        sel = (mag >= edges[i]) & (mag < edges[i + 1])
        n = sel.sum()
        if n < 2:
            continue
        scale = _zp_to_scale(zp[sel])
        med = np.median(scale)
        mad = 1.4826 * np.median(np.abs(scale - med))
        bin_scale.append(med)
        bin_mad.append(max(mad, 1e-6 * med))
        bin_n.append(n)
        bin_centers.append(0.5 * (edges[i] + edges[i + 1]))

    if len(bin_scale) < 2:
        return _global()

    bin_scale = np.array(bin_scale)
    bin_mad = np.array(bin_mad)
    bin_n = np.array(bin_n)
    se = bin_mad / np.sqrt(bin_n)
    weights = 1.0 / se ** 2
    combined_scale = np.sum(weights * bin_scale) / np.sum(weights)
    zp_ab = _scale_to_zp(combined_scale)
    zp_scatter = float(np.average(bin_mad / bin_scale, weights=bin_n)) * 1.0857
    bins = pd.DataFrame({'mag_center': bin_centers, 'scale': bin_scale,
                          'mad': bin_mad, 'n': bin_n})
    return zp_ab, zp_scatter, bins


def _select_isolated(ra, dec, mag, ra_all, dec_all, mag_all,
                      iso_radius_pix=4.0, pix_scale=21.0, delta_mag=2.0):
    """Boolean mask of candidates with no brighter Gaia/catalogue neighbour
    within `iso_radius_pix` (converted to arcsec via `pix_scale`).

    Uses a flat-sky approximation (valid at these small separations), same as
    TESSELLATE's `_select_isolated`.
    """
    iso_arcsec = iso_radius_pix * pix_scale
    ra = np.asarray(ra); dec = np.asarray(dec); mag = np.asarray(mag)
    ra_all = np.asarray(ra_all); dec_all = np.asarray(dec_all); mag_all = np.asarray(mag_all)
    cosdec = np.cos(np.deg2rad(dec_all))

    keep = np.ones(len(ra), dtype=bool)
    for i in range(len(ra)):
        dra = (ra_all - ra[i]) * cosdec
        ddec = dec_all - dec[i]
        sep = np.hypot(dra, ddec) * 3600.0
        neighbour = (sep > 0.5) & (sep < iso_arcsec) & (mag_all < mag[i] + delta_mag)
        if neighbour.any():
            keep[i] = False
    return keep


def _err_keep_mask(e_flux, flux, max_err_factor=3.0):
    """Quality gate on formal per-star flux error, replacing the coarse
    mag<=tmag+1 sanity check used by tessreduce's legacy field_calibrate()."""
    frac_err = np.abs(e_flux / np.where(flux > 0, flux, np.nan))
    med = np.nanmedian(frac_err)
    return np.isfinite(frac_err) & (frac_err < max_err_factor * med) & (flux > 0)


def _star_stamp(image, xpix, ypix, stamp_size):
    half = stamp_size // 2
    xi, yi = int(round(xpix)), int(round(ypix))
    ny, nx = image.shape
    if xi - half < 0 or yi - half < 0 or xi + half + 1 > nx or yi + half + 1 > ny:
        return None, None, None
    stamp = image[yi - half:yi + half + 1, xi - half:xi + half + 1]
    return stamp, xpix - xi, ypix - yi


def _fit_star(image, prf, xpix, ypix, stamp_size, poly_order,
               neighbour_xy=None, flux_bounds=None):
    """Single-star scene fit on a single image (the calibration reference
    frame), optionally deblending catalogue neighbours. Returns
    (flux, e_flux) or (None, None) if the stamp is unusable."""
    stamp, x_sub, y_sub = _star_stamp(image, xpix, ypix, stamp_size)
    if stamp is None or not np.all(np.isfinite(stamp)):
        return None, None
    cent = (stamp_size - 1) / 2.0

    neighbour_dxdy = []
    if neighbour_xy:
        xi, yi = int(round(xpix)), int(round(ypix))
        for nx, ny in neighbour_xy:
            neighbour_dxdy.append((nx - xi, ny - yi))

    A, info = sp.build_design_matrix(prf, cent, (x_sub, y_sub), neighbour_dxdy, stamp_size,
                                      poly_order=poly_order, include_psf_derivatives=False)
    bounds = None
    if flux_bounds is not None:
        lo = np.full(A.shape[1], -np.inf)
        hi = np.full(A.shape[1], np.inf)
        lo[0], hi[0] = flux_bounds
        bounds = (lo, hi)
    coeffs, cov, dof = sp.fit_scene_frame(A, stamp.ravel(), flux_bounds=bounds)
    flux = coeffs[0]
    e_flux = np.sqrt(max(cov[0, 0], 0.0))
    return flux, e_flux


def _calibrate_common(tess, ra, dec, mag, mag_col_label,
                       mag_lo, mag_hi, iso_radius_pix=4.0, stamp_size=9,
                       poly_order=2, refine_iter=3, refine_tol=1e-3,
                       var_tol_mag=0.5, max_err_factor=3.0, max_zp_err=0.1,
                       zp_bin_width=0.5, zp_bin_min_n=5, edge_margin=5):
    """Shared calibration engine: star selection + scene-fit photometry +
    robust combine + iterative refinement, fed either PS1/SkyMapper `tmag`
    (already extinction-corrected & AB) or Gaia `rp_ab`.

    `ra`, `dec`, `mag` are the FULL catalogue (for isolation checks and
    neighbour deblending); the calibration sample is the `mag_lo<mag<mag_hi`
    subset of the same arrays.
    """
    x_all, y_all = tess.wcs.all_world2pix(ra, dec, 0)
    ref = np.asarray(tess.ref, dtype=float)
    ny, nx = ref.shape[-2:]

    sel_range = (mag > mag_lo) & (mag < mag_hi)
    edge_ok = ((x_all > edge_margin) & (x_all < nx - edge_margin)
               & (y_all > edge_margin) & (y_all < ny - edge_margin))
    cal_idx = np.where(sel_range & edge_ok & np.isfinite(x_all) & np.isfinite(y_all))[0]
    if len(cal_idx) < 10:
        zp_ab, zp_err = _zp_fallback(f'too few {mag_col_label} calibration candidates ({len(cal_idx)})')
        return zp_ab, zp_err, None

    iso_keep = _select_isolated(ra[cal_idx], dec[cal_idx], mag[cal_idx], ra, dec, mag,
                                 iso_radius_pix=iso_radius_pix)
    iso_idx = cal_idx[iso_keep]
    if len(iso_idx) < 10:
        zp_ab, zp_err = _zp_fallback(f'too few isolated {mag_col_label} stars ({len(iso_idx)})')
        return zp_ab, zp_err, None

    prf_cam, prf_ccd, prf_sector = tess.tpf.camera, tess.tpf.ccd, tess.tpf.sector

    def _prf_at(xpix, ypix):
        col = int(np.clip(tess.tpf.column - int(tess.size//2) + xpix + 45, 45, 2090))
        row = int(np.clip(tess.tpf.row - int(tess.size//2) + ypix + 1, 1, 2040))
        return sp._prf_cache(prf_cam, prf_ccd, prf_sector, col, row, tess._prf_path)

    # ---- stage 1: isolated-star fits, no neighbour deblending needed ----
    fluxes, e_fluxes, mags, xs, ys = [], [], [], [], []
    for i in iso_idx:
        prf = _prf_at(x_all[i], y_all[i])
        flux, e_flux = _fit_star(ref, prf, x_all[i], y_all[i], stamp_size, poly_order)
        if flux is None:
            continue
        fluxes.append(flux); e_fluxes.append(e_flux); mags.append(mag[i])
        xs.append(x_all[i]); ys.append(y_all[i])

    fluxes = np.array(fluxes); e_fluxes = np.array(e_fluxes); mags = np.array(mags)
    if len(fluxes) < 10:
        zp_ab, zp_err = _zp_fallback(f'too few successful {mag_col_label} stage-1 fits ({len(fluxes)})')
        return zp_ab, zp_err, None

    keep = _err_keep_mask(e_fluxes, fluxes, max_err_factor=max_err_factor)
    if keep.sum() < 10:
        zp_ab, zp_err = _zp_fallback(f'too few good-quality {mag_col_label} stage-1 fits ({keep.sum()})')
        return zp_ab, zp_err, None

    zp_star = mags[keep] + 2.5 * np.log10(fluxes[keep])
    zp_ab, zp_err, bins = _binned_zeropoint(mags[keep], zp_star,
                                             bin_width=zp_bin_width, min_bin_n=zp_bin_min_n)
    if not np.isfinite(zp_ab):
        zp_ab, zp_err = _zp_fallback(f'{mag_col_label} stage-1 zeropoint combine failed')
        return zp_ab, zp_err, None

    if zp_err is not None and zp_err > max_zp_err:
        # stage-2 iterative refinement: re-fit ALL in-range stars (not just
        # isolated ones), deblending catalogue neighbours, with flux bounded
        # around the current zeropoint-predicted value.
        for _ in range(refine_iter):
            fluxes2, e_fluxes2, mags2 = [], [], []
            tol = max(3 * zp_err, var_tol_mag)
            for i in cal_idx:
                neighbour_sel = (np.hypot(x_all - x_all[i], y_all - y_all[i]) < stamp_size) & (np.arange(len(x_all)) != i)
                neighbour_xy = list(zip(x_all[neighbour_sel], y_all[neighbour_sel])) if neighbour_sel.any() else None
                expected = 10 ** (-0.4 * (mag[i] - zp_ab))
                bounds = (expected * 10 ** (-0.4 * tol), expected * 10 ** (0.4 * tol))
                prf = _prf_at(x_all[i], y_all[i])
                flux, e_flux = _fit_star(ref, prf, x_all[i], y_all[i], stamp_size, poly_order,
                                          neighbour_xy=neighbour_xy, flux_bounds=bounds)
                if flux is None:
                    continue
                fluxes2.append(flux); e_fluxes2.append(e_flux); mags2.append(mag[i])
            fluxes2 = np.array(fluxes2); e_fluxes2 = np.array(e_fluxes2); mags2 = np.array(mags2)
            if len(fluxes2) < 10:
                break
            keep2 = _err_keep_mask(e_fluxes2, fluxes2, max_err_factor=max_err_factor)
            if keep2.sum() < 10:
                break
            zp_star2 = mags2[keep2] + 2.5 * np.log10(fluxes2[keep2])
            zp_new, zp_err_new, bins = _binned_zeropoint(mags2[keep2], zp_star2,
                                                          bin_width=zp_bin_width, min_bin_n=zp_bin_min_n)
            if not np.isfinite(zp_new):
                break
            converged = abs(zp_new - zp_ab) < refine_tol
            zp_ab, zp_err = zp_new, zp_err_new
            if converged:
                break

    return zp_ab, zp_err, bins


def calibrate_ps1_skymapper(tess, mag_lo=None, mag_hi=None, iso_radius_pix=4.0,
                             stamp_size=9, poly_order=2, plot=False, **kwargs):
    """Pathway 1: PS1 (dec>-30) or SkyMapper (dec<=-30), with the Tonry-locus
    extinction correction and multi-band synthetic-TESS-mag reconstruction
    reused exactly from tessreduce's existing calibration machinery -- both
    already produce an AB-consistent `tmag`.
    """
    if tess.dec < -30:
        table = Get_Catalogue(tess.tpf, Catalog='skymapper')
        system = 'skymapper'
        if table is None:
            zp_ab, zp_err = _zp_fallback('SkyMapper catalogue unavailable')
            tess.cat = None
            return zp_ab, zp_err
    else:
        table = Get_Catalogue(tess.tpf, Catalog='ps1')
        system = 'ps1'

    try:
        ebv, dat = Tonry_reduce(table, plot=plot, system=system)
    except ValueError:
        zp_ab, zp_err = _zp_fallback(f'Tonry extinction fit failed for {system}')
        tess.cat = table
        tess.ebv = 0.0
        return zp_ab, zp_err
    tess.ebv = float(np.atleast_1d(ebv)[0])

    table = PS1_to_TESS_mag(table, ebv=tess.ebv) if system == 'ps1' else SM_to_TESS_mag(table, ebv=tess.ebv)
    x, y = tess.wcs.all_world2pix(table.RAJ2000.values, table.DEJ2000.values, 0)
    table['col'] = x
    table['row'] = y
    tess.cat = table

    mag_lo = 8.5 if mag_lo is None else mag_lo
    mag_hi = 16.0 if mag_hi is None else mag_hi

    zp_ab, zp_err, bins = _calibrate_common(
        tess, table.RAJ2000.values, table.DEJ2000.values, table['tmag'].values, 'tmag',
        mag_lo=mag_lo, mag_hi=mag_hi, iso_radius_pix=iso_radius_pix,
        stamp_size=stamp_size, poly_order=poly_order, **kwargs)
    return zp_ab, zp_err


def calibrate_gaia(tess, mag_lo=11.0, mag_hi=15.5, iso_radius_pix=4.0,
                    stamp_size=9, poly_order=2, plot=False, **kwargs):
    """Pathway 2: Gaia DR3, queried via astroquery (matching tessreduce's
    existing catalogue-fetch convention, not TESSELLATE's duckdb+local-CSV
    approach). Calibrated directly against Gaia Rp with the standard
    Vega->AB offset; no extinction correction (matching TESSELLATE, since
    this pathway doesn't reconstruct a synthetic multi-band TESS magnitude).
    """
    from .catalog_tools import _query_gaia
    c1 = SkyCoord(tess.ra, tess.dec, frame='icrs', unit='deg')
    pix_scale = 21.0
    rad = Angle(np.max(tess.flux.shape[1:]) * pix_scale + 60, 'arcsec')
    # a local catalogue (offline mode) takes precedence; otherwise the shared, cached Gaia
    # query (ESA archive, VizieR fallback) used by the mask and the scene neighbours
    path = getattr(tess, '_catalogue_path', None)
    if path is not None and os.path.isfile(path):
        table = pd.read_csv(path).rename(columns={'ra': 'RA_ICRS', 'dec': 'DE_ICRS'})
    else:
        table = _query_gaia(c1, rad)
    if table is None or len(table) == 0:
        zp_ab, zp_err = _zp_fallback('Gaia DR3 query returned no sources')
        tess.cat = None
        return zp_ab, zp_err

    if 'RPmag' not in table.columns:
        zp_ab, zp_err = _zp_fallback('Gaia DR3 query missing RPmag column')
        tess.cat = None
        return zp_ab, zp_err
    table = table[np.isfinite(table['Gmag']) & (table['Gmag'] < 19)].reset_index(drop=True)
    if len(table) == 0:
        zp_ab, zp_err = _zp_fallback('No Gaia DR3 sources with Gmag<19 in field')
        tess.cat = None
        return zp_ab, zp_err
    table = table[np.isfinite(table['RPmag'])].reset_index(drop=True)
    table['rp_ab'] = table['RPmag'] + GAIA_RP_AB_OFFSET

    x, y = tess.wcs.all_world2pix(table['RA_ICRS'].values, table['DE_ICRS'].values, 0)
    table['col'] = x
    table['row'] = y
    tess.cat = table

    zp_ab, zp_err, bins = _calibrate_common(
        tess, table['RA_ICRS'].values, table['DE_ICRS'].values, table['rp_ab'].values, 'rp_ab',
        mag_lo=mag_lo, mag_hi=mag_hi, iso_radius_pix=iso_radius_pix,
        stamp_size=stamp_size, poly_order=poly_order, **kwargs)
    return zp_ab, zp_err
