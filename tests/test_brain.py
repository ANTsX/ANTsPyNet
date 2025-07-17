import unittest
import ants
import antspynet
import tensorflow as tf

class Test_deep_atropos_version0(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antspynet.deep_atropos(t1)

class Test_deep_atropos_version1(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antspynet.deep_atropos([t1, None, None])

# class Test_cortical_thickness(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         kk = antspynet.cortical_thickness([t1, None, None])

class Test_dkt_version0(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        dkt = antspynet.desikan_killiany_tourville_labeling(t1, version=0)

class Test_dkt_version1(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        dkt = antspynet.desikan_killiany_tourville_labeling(t1, version=1)

class Test_hoa(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        hoa = antspynet.harvard_oxford_atlas_labeling(t1)

class Test_deep_flash(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        df = antspynet.deep_flash(t1)

class Test_hippmapp3r(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        hipp = antspynet.hippmapp3r_segmentation(t1)

# class Test_brain_age(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         age = antspynet.brain_age(t1, number_of_simulations=3, sd_affine=0.01)

class Test_claustrum(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antspynet.claustrum_segmentation(t1)

class Test_hypothalamus(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antspynet.hypothalamus_segmentation(t1)

class Test_cerebellum(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        cereb = antspynet.cerebellum_morphology(t1, compute_thickness_image=False)	
	
# class Test_brain_tumor(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         flair_file = tf.keras.utils.get_file(fname="flair.nii.gz", origin="https://figshare.com/ndownloader/files/42385077")
#         flair = ants.resample_image(ants.image_read(flair_file), (240, 240, 64), use_voxels=True, interp_type=0)
#         t1_file = tf.keras.utils.get_file(fname="t1.nii.gz", origin="https://figshare.com/ndownloader/files/42385071")
#         t1 = ants.resample_image_to_target(ants.image_read(t1_file), flair)
#         t1_contrast_file = tf.keras.utils.get_file(fname="t1_contrast.nii.gz", origin="https://figshare.com/ndownloader/files/42385068")
#         t1_contrast = ants.resample_image_to_target(ants.image_read(t1_contrast_file), flair)
#         t2_file = tf.keras.utils.get_file(fname="t2.nii.gz", origin="https://figshare.com/ndownloader/files/42385074")
#         t2 = ants.resample_image_to_target(ants.image_read(t2_file), flair)
#         bt = antspynet.brain_tumor_segmentation(flair, t1, t1_contrast, t2, patch_stride_length=32)

class Test_mra(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        mra_file = tf.keras.utils.get_file(fname="mra.nii.gz", origin="https://figshare.com/ndownloader/files/46406755")
        mra = ants.image_read(mra_file)
        vessels = antspynet.brain_mra_vessel_segmentation(mra)

class Test_lesion(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1w_with_lesion.nii.gz", origin="https://figshare.com/ndownloader/files/44053868")
        t1 = ants.image_read(t1_file)
        probability_mask = antspynet.lesion_segmentation(t1, do_preprocessing=True)

class Test_inpainting(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1w_with_lesion.nii.gz", origin="https://figshare.com/ndownloader/files/44053868")
        t1 = ants.image_read(t1_file)
        probability_mask = antspynet.lesion_segmentation(t1, do_preprocessing=True)
        lesion_mask = ants.threshold_image(probability_mask, 0.5, 1.1, 1, 0)
        t1_inpainted = antspynet.whole_head_inpainting(t1, roi_mask=lesion_mask, modality="t1", mode="axial")

if __name__ == '__main__':
    unittest.main()