import unittest
import ants
import antspynet
import tensorflow as tf

class Test_mouse_brain_extraction(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        mouse_t2_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309")
        mouse_t2 = ants.image_read(mouse_t2_file)
        mouse_t2_n4 = ants.n4_bias_field_correction(mouse_t2, 
                                                        rescale_intensities=True,
                                                        shrink_factor=2, 
                                                        convergence={'iters': [50, 50, 50, 50], 'tol': 0.0}, 
                                                        spline_param=20)
        mask = antspynet.mouse_brain_extraction(mouse_t2_n4, modality='t2')

class Test_mouse_brain_parcellation(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        mouse_t2_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309")
        mouse_t2 = ants.image_read(mouse_t2_file)
        mouse_t2_n4 = ants.n4_bias_field_correction(mouse_t2, 
                                                        rescale_intensities=True,
                                                        shrink_factor=2, 
                                                        convergence={'iters': [50, 50, 50, 50], 'tol': 0.0}, 
                                                        spline_param=20)
        parc_nick = antspynet.mouse_brain_parcellation(mouse_t2_n4, 
                                                           mask=None, 
                                                           which_parcellation="nick",      
                                                           return_isotropic_output=True)
        parc_tct = antspynet.mouse_brain_parcellation(mouse_t2_n4, 
                                                          mask=None, 
                                                          which_parcellation="tct",      
                                                          return_isotropic_output=True)                                                      

class Test_mouse_cortical_thickness(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        mouse_t2_file = tf.keras.utils.get_file(fname="mouse.nii.gz", origin="https://ndownloader.figshare.com/files/45289309")
        mouse_t2 = ants.image_read(mouse_t2_file)
        mouse_t2_n4 = ants.n4_bias_field_correction(mouse_t2, 
                                                        rescale_intensities=True,
                                                        shrink_factor=2, 
                                                        convergence={'iters': [50, 50, 50, 50], 'tol': 0.0}, 
                                                        spline_param=20)
        kk = antspynet.mouse_cortical_thickness(mouse_t2_n4, 
                                                    mask=None, 
                                                    return_isotropic_output=True)
        
if __name__ == '__main__':
    unittest.main()