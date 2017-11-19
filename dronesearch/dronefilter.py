#!/usr/bin/env python
"""Filters for on-board processing
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import math

import numpy as np
import cv2
import tensorflow as tf
from logzero import logger


class DroneFilter(object):
    __metaclass__ = abc.ABCMeta

    @classmethod
    def factory(cls, **kwargs):
        filter_type = kwargs.pop('type')
        if filter_type == 'tf_mobilenet':
            return TFMobilenetFilter(**kwargs)
        else:
            raise ValueError('Unsupported filter type: {}'.format(filter_type))

    @abc.abstractmethod
    def open():
        pass

    @abc.abstractmethod
    def process(image):
        pass

    @abc.abstractmethod
    def close():
        pass

    @abc.abstractmethod
    def update():
        pass


class TFMobilenetFilter(DroneFilter):
    def __init__(self, name, model_file, input_height, input_width, input_mean,
                 input_std, input_layer, output_layer, label_file,
                 ratio_tile_width, ratio_tile_height):
        self.name = name
        self.model_file = model_file
        self.input_height = input_height
        self.input_width = input_width
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.label_file = label_file
        self.ratio_tile_width = ratio_tile_width
        self.ratio_tile_height = ratio_tile_height
        self._postive_label = '1:positive'
        self._sess = None
        self._input_operation = None
        self._output_operation = None
        self._labels = None
        self._graph = None

    @classmethod
    def _load_graph(cls, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    @classmethod
    def _load_labels(cls, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def _divide_to_tiles(self, im):
        im_h, im_w, _ = im.shape

        tile_w = im_w * self.ratio_tile_width
        tile_h = im_h * self.ratio_tile_height
        tile_w_int, tile_h_int = int(tile_w), int(tile_h)
        grid_width, grid_height = int(1 / self.ratio_tile_width), int(
            1 / self.ratio_tile_height)

        # resize is needed
        resize_to_evenly_divided = False
        evenly_divided_w = tile_w_int * grid_width
        evenly_divided_h = tile_h_int * grid_height
        if evenly_divided_h != im_h or evenly_divided_w != im_w:
            resize_to_evenly_divided = True
        if resize_to_evenly_divided:
            im = cv2.resize(im, (evenly_divided_w, evenly_divided_h))

        # row majored. if tiles are divided into 2x2
        # then the sequence is (0,0), (0,1), (1,0), (1,1)
        # in which 1st index is on x-aix, 2nd index on y-axis
        tiles = []
        tile_w, tile_h = tile_w_int, tile_h_int
        for h_idx in range(0, grid_width):
            for v_idx in range(0, grid_height):
                tile_x = h_idx * tile_w
                tile_y = v_idx * tile_h
                tile_w = min(tile_w, im_w - tile_x)
                tile_h = min(tile_h, im_h - tile_y)
                current_tile = im[tile_y:tile_y + tile_h, tile_x:
                                  tile_x + tile_w]
                tiles.append(current_tile)
        tiles_np = np.asarray(tiles)
        return tiles_np

    def _tf_divide_to_tiles(self, image_tensor):
        # image_shape = tf.shape(image_tensor)
        # im_height = image_shape[1]
        # im_width = image_shape[2]
        # tile = image[:, :tf.cast(width / 2, tf.int32), :tf.cast(
        #     height / 2, tf.int32), :]
        pass

    def _tf_preprocess(self):
        """Construct preprocess graph"""
        image_batch = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
        float_caster = tf.cast(image_batch, tf.float32)
        resized = tf.image.resize_images(float_caster,
                                         [self.input_height, self.input_width])
        normalized = tf.divide(
            tf.subtract(resized, [self.input_mean]), [self.input_std])
        return normalized, image_batch

    def open(self):
        self._graph = self._load_graph(self.model_file)
        with self._graph.as_default():
            self._preprocess_output, self._preprocess_input = (
                self._tf_preprocess())

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self._input_operation = self._graph.get_operation_by_name(input_name)
        self._output_operation = self._graph.get_operation_by_name(output_name)
        self._labels = self._load_labels(self.label_file)
        assert self._postive_label in self._labels
        self._positive_class = self._labels.index(self._postive_label)
        self._sess = tf.Session(graph=self._graph)

    def process(self, image):
        # TODO(@junjuew) check speed on jetson. GPUs are not used to
        # slice image.On sandstorm, _divide_to_tiles on CPU only takes 0.05ms
        tiles = self._divide_to_tiles(image)
        normalized_tiles = self._sess.run(self._preprocess_output, {
            self._preprocess_input: tiles
        })

        predictions = self._sess.run(
            self._output_operation.outputs[0], {
                self._input_operation.outputs[0]: normalized_tiles
            })
        assert predictions.shape[1] == 2
        positive_tiles_indices = np.where(
            np.argmax(predictions, axis=1) == self._positive_class)
        # TODO(@junjuew) return FilterOutput instead
        return positive_tiles_indices
        return tiles[positive_tiles_indices]

    def close(self):
        self._sess.close()

    def update():
        raise NotImplementedError()
