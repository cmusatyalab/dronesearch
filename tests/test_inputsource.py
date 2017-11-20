from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import unittest

import numpy as np
import cv2

import dronesearch.inputsource as inputsource

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class TestOpenCVInputSource(unittest.TestCase):
    def setUp(self):
        self._test_source = inputsource.OpenCVInputSource(
            os.path.join(PROJECT_DIR, 'data/test.mov'))
        self._test_source.open()

    def tearDown(self):
        self._test_source.close()

    def test_read(self):
        """Test image is in RGB"""
        img = self._test_source.read()
        ground_truth = cv2.imread(os.path.join(PROJECT_DIR, 'data/test.jpg'))
        rgb_ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        np.testing.assert_allclose(img[0][0], rgb_ground_truth[0][0], atol=3)


if __name__ == '__main__':
    unittest.main()
