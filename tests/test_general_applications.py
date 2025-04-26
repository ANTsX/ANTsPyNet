import unittest
import ants
import antspynet
import tensorflow as tf

class Test_super_resolution(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        t1_lr = ants.resample_image(t1, (4, 4, 4), use_voxels=False)
        t1_sr = antspynet.mri_super_resolution(t1_lr, expansion_factor=[1,1,2])

class Test_image_assessment(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        qa = antspynet.tid_neural_image_assessment(t1)

class Test_image_assessment2(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        r16 = ants.image_read(ants.get_data("r16"))
        r64 = ants.image_read(ants.get_data("r64"))
        psnr_value = antspynet.psnr(r16, r64)
        ssim_value = antspynet.ssim(r16, r64)

if __name__ == '__main__':
    unittest.main()