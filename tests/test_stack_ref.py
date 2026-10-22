"""
stack_ref on synthetic cubes (no data download): the reference must always be finite.

S40 Cam2 Ccd2 Cut23 (TESSELLATE, 2026-10-22): star-dominated frame noise varies little (sstd 0.13 around 16.2), and the
reference + every frame within 2 d of it were slightly quieter (15.6) -> all below smed - 3 sstd -> nothing stacked.
S42 Cam4 Ccd2 Cut1: every frame within 2 d of the reference in scattered light (noise 10-550 vs 4.8) -> nothing stacked.
"""
import unittest
import numpy as np

import tessreduce as tr

_cls = tr.tessreduce if isinstance(tr.tessreduce, type) else tr.tessreduce.tessreduce


def _make(noise, ref_ind, cadence=200 / 86400, shape=(20, 20), seed=0):
    rng = np.random.default_rng(seed)
    n = len(noise)
    obj = _cls.__new__(_cls)
    pattern = rng.normal(0, 1, shape)              # one pixel pattern scaled per frame: each frame's measured
    obj.flux = 100 + pattern[None] * np.asarray(noise, float)[:, None, None]   # noise is exactly proportional
    obj.mjd = 59400 + np.arange(n) * cadence
    obj.ref_ind = ref_ind
    obj._ref_time_window = 2
    obj.verbose = 0
    return obj


class TestStackRef(unittest.TestCase):

    def test_quiet_reference_stretch(self):
        n = 3900
        noise = np.full(n, 16.2) + np.random.default_rng(1).normal(0, 0.1, n)
        noise[-1000:] = 15.6                    # the quiet stretch the reference sits in, longer than +-2 d
        obj = _make(noise, ref_ind=n - 30)
        obj.stack_ref()
        self.assertTrue(np.isfinite(obj.ref).all())

    def test_window_all_noisy_falls_back(self):
        n = 6000                                # 200 s cadence: +-2 d = +-864 frames
        noise = np.full(n, 4.8) + np.random.default_rng(2).normal(0, 0.1, n)
        noise[2000:4100] = 120.0                # scattered light > 2 d either side of the reference
        obj = _make(noise, ref_ind=3050)
        obj.stack_ref()
        self.assertTrue(np.isfinite(obj.ref).all())
        # the fallback stacks quiet frames: the reference is no noisier than a quiet frame's scatter
        self.assertLess(np.std(obj.ref - 100), 6.0)

    def test_nan_frames_ignored(self):
        n = 2000
        obj = _make(5.0 + np.random.default_rng(3).normal(0, 0.1, n), ref_ind=1000)
        obj.flux[990:1010] = np.nan
        obj.stack_ref()
        self.assertTrue(np.isfinite(obj.ref).all())

    def test_identical_noise(self):
        obj = _make(np.full(500, 5.0), ref_ind=250)  # sstd = 0: frames at the median must still pass
        obj.stack_ref()
        self.assertTrue(np.isfinite(obj.ref).all())

    def test_noisy_frames_still_rejected(self):
        n = 2000
        noise = 5.0 + np.random.default_rng(4).normal(0, 0.1, n)
        noise[1000:1020] = 500.0                # a burst near the reference must not enter the stack
        obj = _make(noise, ref_ind=1050)
        obj.stack_ref()
        self.assertLess(np.std(obj.ref - 100), 5.0)


if __name__ == '__main__':
    unittest.main()
