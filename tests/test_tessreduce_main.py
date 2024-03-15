import unittest
import tessreduce as tr

class TestTESSreduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initial setup run only once before all the tests
        """

        ra = 10.127#189.1385817
        dec = -50.687#11.2316535
        sector = 2
        tess = tr.tessreduce(ra=ra, dec=dec, sector=sector, reduce=False)
        tess.get_ref()
        cls.tess = tess

    def test_Make_mask(self):
        self.tess.make_mask()

    def test_background(self):
        self.tess.background()

    def test_align(self):
        self.tess.fit_shift()

    def test_Shift_images(self):
        self.tess.shift_images()

    def test_field_calibrate(self):
        self.tess.field_calibrate()

    def test_Diff_lc(self):
        self.tess.diff_lc()

    def test_full(self):
        self.tess.reduce()

if __name__ == '__main__':
    unittest.main()
