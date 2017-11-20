from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import unittest

import tensorflow as tf
import numpy as np
import cv2

import dronesearch.dronefilter as dronefilter

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


class TestTFMobilenetFilter(unittest.TestCase):
    def setUp(self):
        model_file = os.path.join(
            PROJECT_DIR,
            'frozen_models/stanford_2_more_test/optimized_frozen_graph.pb')
        label_file = os.path.join(
            PROJECT_DIR, 'frozen_models/stanford_2_more_test/labels.txt')
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        self._input_layer = 'input'
        self._output_layer = 'MobilenetV1/Predictions/Reshape_1'
        ratio_tile_width = 0.25
        ratio_tile_height = 0.25

        self._test_filter = dronefilter.TFMobilenetFilter(
            name='test',
            model_file=model_file,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std,
            input_layer=self._input_layer,
            output_layer=self._output_layer,
            label_file=label_file,
            ratio_tile_width=ratio_tile_width,
            ratio_tile_height=ratio_tile_height)
        self._test_filter.open()

    def tearDown(self):
        self._test_filter.close()

    def test_graph_elements(self):
        """Test the graph has correct nodes"""
        graph_nodes = [
            n.name for n in self._test_filter._graph.as_graph_def().node
        ]
        input_node_name = "import/" + self._input_layer
        output_node_name = "import/" + self._output_layer
        self.assertIn(input_node_name, graph_nodes)
        self.assertIn(output_node_name, graph_nodes)

    def test_process_random_image(self):
        random_image = np.random.rand(224, 224, 3) * 255
        random_image = random_image.astype(np.uint8)
        self._test_filter.process(random_image)

    def test_process_test_image(self):
        image_path = os.path.join(PROJECT_DIR, 'data/test.jpg')
        image_positive_tile_indices = [(1, 2), (2, 0), (2, 1), (2, 3), (3, 1),
                                       (3, 2)]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        outputs = self._test_filter.process(image)
        self.assertEqual(outputs.indices, image_positive_tile_indices)


if __name__ == '__main__':
    unittest.main()
