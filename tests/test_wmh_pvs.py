import unittest
import ants
import antspynet
import tensorflow as tf

class Test_sysu_wmh(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1.nii.gz", origin="https://figshare.com/ndownloader/files/40251796")
        t1 = ants.image_read(t1_file)
        flair_file = tf.keras.utils.get_file(fname="flair.nii.gz", origin="https://figshare.com/ndownloader/files/40251793")
        flair = ants.image_read(flair_file)
        wmh = antspynet.sysu_media_wmh_segmentation(flair, t1)

class Test_hypermapp3r_wmh(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1.nii.gz", origin="https://figshare.com/ndownloader/files/40251796")
        t1 = ants.image_read(t1_file)
        flair_file = tf.keras.utils.get_file(fname="flair.nii.gz", origin="https://figshare.com/ndownloader/files/40251793")
        flair = ants.image_read(flair_file)
        wmh = antspynet.hypermapp3r_segmentation(t1, flair)

class Test_shiva_wmh(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1.nii.gz", origin="https://figshare.com/ndownloader/files/40251796")
        t1 = ants.image_read(t1_file)
        flair_file = tf.keras.utils.get_file(fname="flair.nii.gz", origin="https://figshare.com/ndownloader/files/40251793")
        flair = ants.image_read(flair_file)
        wmh = antspynet.shiva_wmh_segmentation(flair, t1, which_model="all")

class Test_ants_wmh(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="t1.nii.gz", origin="https://figshare.com/ndownloader/files/40251796")
        t1 = ants.image_read(t1_file)
        t1 = ants.resample_image(t1, (240, 240, 64), use_voxels=True)
        flair_file = tf.keras.utils.get_file(fname="flair.nii.gz", origin="https://figshare.com/ndownloader/files/40251793")
        flair = ants.image_read(flair_file)
        flair = ants.resample_image(flair, (240, 240, 64), use_voxels=True)
        wmh = antspynet.wmh_segmentation(flair, t1, use_combined_model=True)

class Test_shiva_wmh(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1_file = tf.keras.utils.get_file(fname="pvs_t1.nii.gz", origin="https://figshare.com/ndownloader/files/48675367")
        t1 = ants.image_read(t1_file)
        flair_file = tf.keras.utils.get_file(fname="pvs_flair.nii.gz", origin="https://figshare.com/ndownloader/files/48675352")
        flair = ants.image_read(flair_file)
        pvs = antspynet.shiva_pvs_segmentation(t1, flair, which_model = "all")

if __name__ == '__main__':
    unittest.main()