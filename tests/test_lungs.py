import unittest
import ants
import antspynet
import tensorflow as tf

class Test_lung_extraction_ct(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        ct_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934234")
        ct = ants.image_read(ct_file)
        lung_ex = antspynet.lung_extraction(ct, modality="ct")

class Test_lung_extraction_proton(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        proton_file = tf.keras.utils.get_file(fname="protonLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934228")
        proton = ants.image_read(proton_file)
        lung_ex = antspynet.lung_extraction(proton, modality="proton")
        lung_ex = antspynet.lung_extraction(proton, modality="protonLobes")
        lung_mask = ants.threshold_image(lung_ex['segmentation_image'], 0, 0, 0, 1 )
        lung_ex = antspynet.lung_extraction(lung_mask, modality="maskLobes")

class Test_lung_extraction_cxr(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        cxr_file = tf.keras.utils.get_file(fname="cxr.nii.gz", origin="https://figshare.com/ndownloader/files/42934237")
        cxr = ants.image_read(cxr_file)
        lung_ex = antspynet.lung_extraction(cxr, modality="xray")

class Test_lung_extraction_ventilation(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        ventilation_file = tf.keras.utils.get_file(fname="ventilationLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934231")
        ventilation = ants.image_read(ventilation_file)
        lung_ex = antspynet.lung_extraction(ventilation, modality="ventilation")

class Test_elbicho(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        proton_file = tf.keras.utils.get_file(fname="protonLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934228")
        proton = ants.image_read(proton_file)
        lung_ex = antspynet.lung_extraction(proton, modality="proton")
        lung_mask = ants.threshold_image(lung_ex['segmentation_image'], 0, 0, 0, 1 )
        ventilation_file = tf.keras.utils.get_file(fname="ventilationLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934231")
        ventilation = ants.image_read(ventilation_file)
        eb = antspynet.el_bicho(ventilation, lung_mask)

# class Test_ct_arteries(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         ct_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934234")
#         ct = ants.image_read(ct_file)
#         arteries = antspynet.lung_pulmonary_artery_segmentation(ct)

# class Test_ct_airways(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         ct_file = tf.keras.utils.get_file(fname="ctLung.nii.gz", origin="https://figshare.com/ndownloader/files/42934234")
#         ct = ants.image_read(ct_file)
#         airways = antspynet.lung_airway_segmentation(ct)

class Test_chexnet(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        cxr_file = tf.keras.utils.get_file(fname="cxr.nii.gz", origin="https://figshare.com/ndownloader/files/42934237")
        cxr = ants.image_read(cxr_file)
        antspynet.chexnet(cxr, use_antsxnet_variant=False)
        antspynet.chexnet(cxr, use_antsxnet_variant=True, include_tuberculosis_diagnosis=True)

if __name__ == '__main__':
    unittest.main()