import unittest
import ants
import antspynet

class Test_t1(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antspynet.brain_extraction(t1, modality="t1")

# class Test_t1nobrainer(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         seg = antspynet.brain_extraction(t1, modality="t1nobrainer")

# class Test_t1combined(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         seg = antspynet.brain_extraction(t1, modality="t1combined")

# class Test_t1threetissue(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antspynet.brain_extraction(t1, modality="t1threetissue")

# class Test_t1hemi(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antspynet.brain_extraction(t1, modality="t1hemi")

# class Test_t1lobes(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antspynet.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antspynet.brain_extraction(t1, modality="t1lobes")

if __name__ == '__main__':
    unittest.main()