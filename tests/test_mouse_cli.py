import os
import tempfile
import unittest
from unittest.mock import patch

import ants
import numpy as np

from antspynet.cli.mouse_brain_extraction import main


class TestMouseBrainExtractionCli(unittest.TestCase):
    def test_main_routes_to_mouse_brain_extraction(self):
        input_image = ants.from_numpy(np.ones((4, 4, 4)))
        output_image = ants.from_numpy(np.full((4, 4, 4), 0.25))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            output_path = os.path.join(tmpdir, "output.nii.gz")

            calls = {}

            def fake_image_read(path):
                calls["image_read"] = path
                return input_image

            def fake_mouse_brain_extraction(
                image,
                modality,
                return_isotropic_output,
                which_axis,
                verbose,
            ):
                calls["mouse_brain_extraction"] = {
                    "image": image,
                    "modality": modality,
                    "return_isotropic_output": return_isotropic_output,
                    "which_axis": which_axis,
                    "verbose": verbose,
                }
                return output_image

            def fake_image_write(image, path):
                calls["image_write"] = {"image": image, "path": path}

            with patch("antspynet.cli.mouse_brain_extraction.ants.image_read", side_effect=fake_image_read), \
                 patch("antspynet.cli.mouse_brain_extraction.antspynet.mouse_brain_extraction", side_effect=fake_mouse_brain_extraction), \
                 patch("antspynet.cli.mouse_brain_extraction.ants.image_write", side_effect=fake_image_write):
                exit_code = main([
                    input_path,
                    output_path,
                    "--modality", "t2",
                    "--isotropic-output",
                    "--axis", "1",
                    "--verbose",
                ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(calls["image_read"], input_path)
        self.assertEqual(calls["image_write"]["path"], output_path)
        self.assertIs(calls["image_write"]["image"], output_image)
        self.assertEqual(calls["mouse_brain_extraction"]["modality"], "t2")
        self.assertTrue(calls["mouse_brain_extraction"]["return_isotropic_output"])
        self.assertEqual(calls["mouse_brain_extraction"]["which_axis"], 1)
        self.assertTrue(calls["mouse_brain_extraction"]["verbose"])


if __name__ == "__main__":
    unittest.main()