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

# Gaia DR3 Vega -> AB offsets, ZP_AB - ZP_VEGA from the published passband zero points (Riello et al. 2021,
# Table 3): G_RP 25.1040 - 24.7479, G_BP 25.3540 - 25.3385. A synthetic value of 0.379 (pysynphot Vega through
# the Rp passband) was used previously; it omits Gaia's VEGAMAG convention that Vega itself is 0.023 mag, so it made
# every Gaia-pathway zeropoint 0.023 mag too bright. The older literature value 0.152 was ~0.2 mag too small.
GAIA_RP_AB_OFFSET = 25.1040 - 24.7479
GAIA_BP_AB_OFFSET = 25.3540 - 25.3385

_GAIA_COLOUR = None


def gaia_tess_mag(rp_vega, bp_vega):
    """TESS AB magnitude of a star from its Gaia DR3 G_RP and G_BP (VEGAMAG): G_RP,AB plus the colour correction
    TESS_AB - G_RP,AB = S((G_BP - G_RP)_AB), a cubic smoothing spline fitted to synthetic photometry of Calspec
    spectra and Castelli-Kurucz models (packaged table gaia_rp_tess_colour.csv). Stars outside the tabulated colour
    range, or without G_BP, get NaN: the correction is not extrapolated."""
    global _GAIA_COLOUR
    if _GAIA_COLOUR is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gaia_rp_tess_colour.csv')
        _GAIA_COLOUR = pd.read_csv(path, comment='#').values
    rp_ab = np.asarray(rp_vega, float) + GAIA_RP_AB_OFFSET
    colour = np.asarray(bp_vega, float) + GAIA_BP_AB_OFFSET - rp_ab
    grid, corr = _GAIA_COLOUR[:, 0], _GAIA_COLOUR[:, 1]
    inside = np.isfinite(colour) & (colour >= grid[0]) & (colour <= grid[-1])
    out = np.full(rp_ab.shape, np.nan)
    out[inside] = rp_ab[inside] + np.interp(colour[inside], grid, corr) / 1e3
    return out

_ZP_FALLBACK = 20.6
_ZP_FALLBACK_ERR = 0.1
# accuracy of the zeropoint map for one field: rms difference between the map and 7 fields calibrated from
# their own stars (SNe 2020adw, cdj, fqv, yzo, GRB 191016A, SDSS J1227, TOI-519 S7); the map's own
# sector-to-sector scatter (~0.007 mag) is far smaller, so the field-to-field term dominates
_ZP_MAP_ACCURACY = 0.10


def _zp_to_scale(zp):
    """AB zeropoint (mag) -> linear flux-scale factor."""
    return 10 ** (0.4 * np.asarray(zp, dtype=float))


def _scale_to_zp(f):
    """Linear flux-scale factor -> AB zeropoint (mag)."""
    return 2.5 * np.log10(f)


_ZP_MAP = None


def zp_from_map(camera, ccd, column, row):
    """Typical AB zeropoint at a CCD position, from the packaged map of TESSELLATE zeropoints (median over
    Sectors 27-55 for each of the 8x8 cuts of every CCD, corrected to the Gaia Rp Vega->AB offset of Riello et al. 2021),
    interpolated with a bicubic spline through the cut centres, which lie on a rectangular grid; beyond the
    outermost centres (the last ~130 px of each CCD edge) the spline continues the local trend.
    Returns (zp_ab, zp_scatter), the scatter being the sector-to-sector spread of that part of the CCD, or
    (nan, nan) if the map is unavailable."""
    global _ZP_MAP
    from scipy.interpolate import RectBivariateSpline
    if _ZP_MAP is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessellate_zp_map.csv')
        _ZP_MAP = pd.read_csv(path, comment='#') if os.path.exists(path) else False
    if _ZP_MAP is False:
        return np.nan, np.nan
    m = _ZP_MAP[(_ZP_MAP.cam == camera) & (_ZP_MAP.ccd == ccd)]
    if len(m) == 0:
        return np.nan, np.nan
    cols = np.sort(m.column.unique())
    rows = np.sort(m.row.unique())
    if len(cols) < 4 or len(rows) < 4 or len(m) != len(cols) * len(rows):
        return np.nan, np.nan
    out = []
    for key in ('zp_ab', 'zp_scatter'):
        grid = m.pivot(index='column', columns='row', values=key).loc[cols, rows].values
        out.append(float(RectBivariateSpline(cols, rows, grid, kx=3, ky=3)(column, row)[0, 0]))
    return out[0], max(out[1], 0.0)


def _zp_fallback(tess, reason):
    """Zeropoint when the field calibration fails: the packaged focal-plane map at the cutout centre, else 20.6."""
    zp, err = np.nan, np.nan
    try:
        # the target's CCD position from the sector WCS: TESScut files report column/row as 0
        import logging
        import tesswcs
        level = tesswcs.log.level
        tesswcs.log.setLevel(logging.ERROR)
        try:
            w = tesswcs.WCS.from_sector(int(tess.sector), int(tess.camera), int(tess.ccd))
            x, y = w.world_to_pixel(SkyCoord(tess.ra, tess.dec, unit='deg'))
        finally:
            tesswcs.log.setLevel(level)
        # tesswcs gives 0-based FFI array pixels; the map uses array index + 1
        zp, err = zp_from_map(int(tess.camera), int(tess.ccd), float(x) + 1, float(y) + 1)
    except Exception:
        pass
    if np.isfinite(zp):
        err = float(np.hypot(err, _ZP_MAP_ACCURACY))
        warnings.warn(f'field_calibration: {reason} -- falling back to the zeropoint map, zp_ab={zp:.3f}+-{err:.3f}')
        return zp, err
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
               neighbour_xy=None, flux_bounds=None, neighbour_weights=None):
    """Single-star scene fit on a single image (the calibration reference
    frame), optionally deblending catalogue neighbours. With
    `neighbour_weights` (catalogue flux of each neighbour relative to the
    target) the neighbours are rendered together as one scene with a single
    free scale, rather than one free flux each: at 21 arcsec per pixel a
    crowded stamp holds more catalogue sources than pixels. Returns
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

    if neighbour_weights is not None:
        A, info = sp.build_design_matrix(prf, cent, (x_sub, y_sub), [], stamp_size,
                                          poly_order=poly_order, include_psf_derivatives=False)
        if neighbour_dxdy:
            A = np.column_stack([A, sp.render_catalogue_scene(prf, stamp_size, neighbour_dxdy, neighbour_weights)])
    else:
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
                       zp_bin_width=0.5, zp_bin_min_n=5, edge_margin=5, cal_ok=None):
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
    if cal_ok is not None:
        # stars whose catalogue magnitude is not a valid TESS magnitude (e.g. outside the colour-correction
        # range) still enter as neighbours but are not calibration stars
        sel_range &= np.asarray(cal_ok, bool)
    edge_ok = ((x_all > edge_margin) & (x_all < nx - edge_margin)
               & (y_all > edge_margin) & (y_all < ny - edge_margin))
    cal_idx = np.where(sel_range & edge_ok & np.isfinite(x_all) & np.isfinite(y_all))[0]
    if len(cal_idx) < 10:
        zp_ab, zp_err = _zp_fallback(tess, f'too few {mag_col_label} calibration candidates ({len(cal_idx)})')
        return zp_ab, zp_err, None

    iso_keep = _select_isolated(ra[cal_idx], dec[cal_idx], mag[cal_idx], ra, dec, mag,
                                 iso_radius_pix=iso_radius_pix)
    iso_idx = cal_idx[iso_keep]

    prf_cam, prf_ccd, prf_sector = tess.camera, tess.ccd, tess.sector

    def _prf_at(xpix, ypix):
        col = int(np.clip(tess.column - int(tess.size//2) + xpix + 45, 45, 2090))
        row = int(np.clip(tess.row - int(tess.size//2) + ypix + 1, 1, 2040))
        return sp._prf_cache(prf_cam, prf_ccd, prf_sector, col, row, tess._prf_path)

    def _deblended_zeropoint(zp_prior=None, tol=None):
        # scene fit of every in-range star with its catalogue neighbours as extra PRF columns;
        # with a prior zeropoint the target flux is bounded around its predicted value
        fl, ef, mg = [], [], []
        reach = (stamp_size - 1) / 2.0 + 3  # stamp plus the rendering pad: sources whose wings fall in the stamp
        for i in cal_idx:
            neighbour_sel = ((np.abs(x_all - x_all[i]) <= reach) & (np.abs(y_all - y_all[i]) <= reach)
                             & (np.arange(len(x_all)) != i) & np.isfinite(mag))
            neighbour_xy = list(zip(x_all[neighbour_sel], y_all[neighbour_sel])) if neighbour_sel.any() else None
            weights = 10 ** (-0.4 * (mag[neighbour_sel] - mag[i]))
            bounds = None
            if zp_prior is not None:
                expected = 10 ** (-0.4 * (mag[i] - zp_prior))
                bounds = (expected * 10 ** (-0.4 * tol), expected * 10 ** (0.4 * tol))
            prf = _prf_at(x_all[i], y_all[i])
            flux, e_flux = _fit_star(ref, prf, x_all[i], y_all[i], stamp_size, poly_order,
                                      neighbour_xy=neighbour_xy, flux_bounds=bounds, neighbour_weights=weights)
            if flux is None:
                continue
            fl.append(flux); ef.append(e_flux); mg.append(mag[i])
        fl = np.array(fl); ef = np.array(ef); mg = np.array(mg)
        if len(fl) < 10:
            return np.nan, np.nan, None, len(fl)
        k = _err_keep_mask(ef, fl, max_err_factor=max_err_factor) & (fl > 0)
        if k.sum() < 10:
            return np.nan, np.nan, None, int(k.sum())
        zp_star = mg[k] + 2.5 * np.log10(fl[k])
        z, ze, b = _binned_zeropoint(mg[k], zp_star, bin_width=zp_bin_width, min_bin_n=zp_bin_min_n)
        return z, ze, b, int(k.sum())

    def _refine(zp_ab, zp_err, bins):
        # stage-2 iterative refinement: re-fit ALL in-range stars (not just isolated ones),
        # deblending catalogue neighbours, with flux bounded around the current zeropoint
        if not (zp_err is not None and zp_err > max_zp_err):
            return zp_ab, zp_err, bins
        for _ in range(refine_iter):
            z, ze, b, _n = _deblended_zeropoint(zp_ab, max(3 * zp_err, var_tol_mag))
            if not np.isfinite(z):
                break
            converged = abs(z - zp_ab) < refine_tol
            zp_ab, zp_err, bins = z, ze, b
            if converged:
                break
        return zp_ab, zp_err, bins

    if len(iso_idx) < 10:
        # too few isolated stars (small cutout or crowded field): start from unbounded
        # deblended fits of every calibration star, then refine as below
        zp_ab, zp_err, bins, n_fit = _deblended_zeropoint()
        if not np.isfinite(zp_ab):
            zp_ab, zp_err = _zp_fallback(tess, f'too few isolated {mag_col_label} stars ({len(iso_idx)}) '
                                         f'and too few deblended fits ({n_fit})')
            return zp_ab, zp_err, None
        return _refine(zp_ab, zp_err, bins)

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
        zp_ab, zp_err = _zp_fallback(tess, f'too few successful {mag_col_label} stage-1 fits ({len(fluxes)})')
        return zp_ab, zp_err, None

    keep = _err_keep_mask(e_fluxes, fluxes, max_err_factor=max_err_factor)
    if keep.sum() < 10:
        zp_ab, zp_err = _zp_fallback(tess, f'too few good-quality {mag_col_label} stage-1 fits ({keep.sum()})')
        return zp_ab, zp_err, None

    zp_star = mags[keep] + 2.5 * np.log10(fluxes[keep])
    zp_ab, zp_err, bins = _binned_zeropoint(mags[keep], zp_star,
                                             bin_width=zp_bin_width, min_bin_n=zp_bin_min_n)
    if not np.isfinite(zp_ab):
        zp_ab, zp_err = _zp_fallback(tess, f'{mag_col_label} stage-1 zeropoint combine failed')
        return zp_ab, zp_err, None

    return _refine(zp_ab, zp_err, bins)


def calibrate_ps1_skymapper(tess, mag_lo=None, mag_hi=None, iso_radius_pix=4.0,
                             stamp_size=9, poly_order=2, plot=False, **kwargs):
    """Pathway 1: PS1 (dec>-30) or SkyMapper (dec<=-30), with the Tonry-locus
    extinction correction and multi-band synthetic-TESS-mag reconstruction
    reused exactly from tessreduce's existing calibration machinery -- both
    already produce an AB-consistent `tmag`.
    """
    if tess.dec < -30:
        table = Get_Catalogue(tess.ra, tess.dec, tess.flux.shape, Catalog='skymapper')
        system = 'skymapper'
        if table is None:
            zp_ab, zp_err = _zp_fallback(tess, 'SkyMapper catalogue unavailable')
            tess.cat = None
            return zp_ab, zp_err
    else:
        table = Get_Catalogue(tess.ra, tess.dec, tess.flux.shape, Catalog='ps1')
        system = 'ps1'

    try:
        ebv, dat = Tonry_reduce(table, plot=plot, system=system)
    except ValueError:
        zp_ab, zp_err = _zp_fallback(tess, f'Tonry extinction fit failed for {system}')
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
        zp_ab, zp_err = _zp_fallback(tess, 'Gaia DR3 query returned no sources')
        tess.cat = None
        return zp_ab, zp_err

    if 'RPmag' not in table.columns:
        zp_ab, zp_err = _zp_fallback(tess, 'Gaia DR3 query missing RPmag column')
        tess.cat = None
        return zp_ab, zp_err
    table = table[np.isfinite(table['Gmag']) & (table['Gmag'] < 19)].reset_index(drop=True)
    if len(table) == 0:
        zp_ab, zp_err = _zp_fallback(tess, 'No Gaia DR3 sources with Gmag<19 in field')
        tess.cat = None
        return zp_ab, zp_err
    table = table[np.isfinite(table['RPmag'])].reset_index(drop=True)
    if 'BPmag' in table.columns:
        # TESS AB magnitude from G_RP and the G_BP - G_RP colour correction; stars outside the calibrated
        # colour range (or without G_BP) keep NaN and so stay out of the calibration sample, but remain in
        # the catalogue as neighbours (their G_RP AB magnitude is a close enough weight for deblending)
        table['rp_ab'] = gaia_tess_mag(table['RPmag'].values, table['BPmag'].values)
        no_colour = ~np.isfinite(table['rp_ab'])
        table.loc[no_colour, 'rp_ab'] = table.loc[no_colour, 'RPmag'] + GAIA_RP_AB_OFFSET
        table['colour_ok'] = ~no_colour
    else:
        warnings.warn('field_calibration: no G_BP in the Gaia catalogue -- using G_RP without the colour correction')
        table['rp_ab'] = table['RPmag'] + GAIA_RP_AB_OFFSET
        table['colour_ok'] = True

    x, y = tess.wcs.all_world2pix(table['RA_ICRS'].values, table['DE_ICRS'].values, 0)
    table['col'] = x
    table['row'] = y
    tess.cat = table

    zp_ab, zp_err, bins = _calibrate_common(
        tess, table['RA_ICRS'].values, table['DE_ICRS'].values, table['rp_ab'].values, 'rp_ab',
        mag_lo=mag_lo, mag_hi=mag_hi, iso_radius_pix=iso_radius_pix,
        stamp_size=stamp_size, poly_order=poly_order, cal_ok=table['colour_ok'].values, **kwargs)
    return zp_ab, zp_err
