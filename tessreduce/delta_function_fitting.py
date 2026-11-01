import numpy as np


def _shifted(reference, size):
	"""Stack of the reference shifted by every (dy, dx) in a size x size box; edges that wrap are set to NaN."""
	h = size // 2
	ny, nx = reference.shape
	out = np.full((size * size, ny, nx), np.nan)
	k = 0
	for dy in range(-h, h + 1):
		for dx in range(-h, h + 1):
			out[k, max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)] = \
				reference[max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)]
			k += 1
	return out


def match_kernels(images, reference, fit_pix, size=5, clip=5, weights=None):
	"""
	Delta-function basis kernel matching (Bramich 2008). For each frame the kernel K and a constant b minimise
	sum w (image - reference*K - b)^2 over the fit pixels, which is linear in the size*size kernel values.
	The design matrix is the same for every frame, so all frames are solved together; pixels with a residual
	beyond `clip` sigma in a frame (variable stars, cosmic rays) are rejected and that frame is re-solved.

	Parameters
	----------
	images : array (n, ny, nx)
		Frames aligned to the reference.
	reference : array (ny, nx)
	fit_pix : bool array (ny, nx)
		Pixels that constrain the kernel: stars in the reference, away from the target and the edges.
	size : int
		Kernel width in pixels (odd).
	weights : array (ny, nx), optional
		Pixel weights. The default is uniform: the PSF mismatch is largest on bright stars, which Poisson
		weighting suppresses (on SN 2020kyx S25 it left the orbit-2 star residuals unchanged).

	Returns
	-------
	diff : array (n, ny, nx)
		image - reference*K - b.
	kernels : array (n, size, size)
	bkg : array (n,)
	"""
	images = np.asarray(images, dtype=float)
	B = _shifted(reference, size)
	pix = fit_pix & np.all(np.isfinite(B), axis=0) & np.isfinite(reference)
	if weights is None:
		weights = np.ones_like(reference)
	sw = np.sqrt(weights[pix])
	A = np.column_stack([B[:, pix].T, np.ones(pix.sum())])
	Aw = A * sw[:, None]
	Y = images[:, pix].T
	n = len(images)
	coef = np.full((n, A.shape[1]), np.nan)
	good = np.all(np.isfinite(Y), axis=0)
	if good.any():
		coef[good] = np.linalg.lstsq(Aw, Y[:, good] * sw[:, None], rcond=None)[0].T
	for i in np.where(~good)[0]:
		ok = np.isfinite(Y[:, i])
		if ok.sum() > 3 * A.shape[1]:
			coef[i] = np.linalg.lstsq(Aw[ok], Y[ok, i] * sw[ok], rcond=None)[0]
	# one clipping pass, frame by frame, only for frames with outlying fit pixels
	res = (Y - A @ np.nan_to_num(coef).T) * sw[:, None]
	sig = 1.4826 * np.nanmedian(np.abs(res), axis=0)
	bad = np.abs(res) > clip * sig
	for i in np.where(bad.any(axis=0) & np.isfinite(coef).all(axis=1))[0]:
		ok = np.isfinite(Y[:, i]) & ~bad[:, i]
		if ok.sum() > 3 * A.shape[1]:
			coef[i] = np.linalg.lstsq(Aw[ok], Y[ok, i] * sw[ok], rcond=None)[0]
	kernels = coef[:, :-1].reshape(n, size, size)
	bkg = coef[:, -1]
	# shifts that run off the edge reuse the unshifted reference there
	Bf = np.where(np.isfinite(B), B, np.nan_to_num(reference))
	template = np.einsum('nk,kyx->nyx', np.nan_to_num(coef[:, :-1]), Bf) + np.nan_to_num(bkg)[:, None, None]
	diff = images - template
	diff[~np.isfinite(coef).all(axis=1)] = np.nan
	return diff, kernels, bkg
