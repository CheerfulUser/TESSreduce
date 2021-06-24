import unittest
import tessreduce as tr

class TestTESSreduce(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Initial setup run only once before all the tests
        """

        ra = 189.1385817
        dec = 11.2316535
        tess = tr.tessreduce(ra=ra, dec=dec)
        tess.get_ref()
        cls.tess = tess

    def test_Make_mask(self):
        self.tess.make_mask()

    def test_background(self):
        self.tess.background()

    def test_Centroids_DAO(self):
        self.tess.centroids_DAO()

    def test_Shift_images(self):
        self.tess.shift_images()

    def test_field_calibrate(self):
        self.tess.field_calibrate()

    def test_Diff_lc(self):
        self.tess.diff_lc()

if __name__ == '__main__':
    unittest.main()
