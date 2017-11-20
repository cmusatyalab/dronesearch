#!/usr/bin/env python
"""Filters for on-board processing
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import math
import cPickle as pickle
import time

import numpy as np
import cv2
import tensorflow as tf
from logzero import logger

import dronesearch.utils as utils


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
                 ratio_tile_width, ratio_tile_height, positive_label):
        self.name = name
        self.model_file = model_file
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.input_mean = int(input_mean)
        self.input_std = int(input_std)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.label_file = label_file
        self.ratio_tile_width = float(ratio_tile_width)
        self.ratio_tile_height = float(ratio_tile_height)
        self.positive_label = positive_label
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
        # tiles_np = np.asarray(tiles)
        tiles_np = tiles
        return tiles_np

    def _tile_list_index_to_grid_index(self, index):
        """Convert index in 1d list back into grid index of (x, y) for tiles

        Args:
          index: 

        Returns:

        """
        grid_height = int(1 / self.ratio_tile_height)
        grid_x = int(index / grid_height)
        grid_y = index % grid_height
        return (grid_x, grid_y)

    def _tf_divide_to_tiles(self, image_tensor):
        # tf's implementation of cropping images into tiles
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
        assert self.positive_label in self._labels
        self._positive_class = self._labels.index(self.positive_label)
        self._sess = tf.Session(graph=self._graph)

    @utils.timeit
    def process(self, image):
        # TODO(@junjuew) check speed on jetson. GPUs are not used to
        # slice image.On sandstorm, _divide_to_tiles on CPU only takes 0.05ms
        st = time.time()
        tiles = self._divide_to_tiles(image)
        logger.debug('cropping takes {}ms'.format((time.time() - st) * 1000))
        st = time.time()
        normalized_tiles = self._sess.run(self._preprocess_output, {
            self._preprocess_input: tiles
        })
        logger.debug('preprocess takes {}ms'.format((time.time() - st) * 1000))
        st = time.time()

        predictions = self._sess.run(
            self._output_operation.outputs[0], {
                self._input_operation.outputs[0]: normalized_tiles
            })
        logger.debug('predictions takes {}ms'.format(
            (time.time() - st) * 1000))
        st = time.time()

        assert predictions.shape[1] == 2
        positive_indices = np.where(
            np.argmax(predictions, axis=1) == self._positive_class)[0]
        positive_grid_indices = [
            self._tile_list_index_to_grid_index(positive_indice)
            for positive_indice in positive_indices
        ]
        logger.debug('convert to labels {}ms'.format(
            (time.time() - st) * 1000))
        return TileFilterOutput(positive_grid_indices,
                                [tiles[idx] for idx in positive_indices])

    def close(self):
        self._sess.close()

    def update():
        raise NotImplementedError()


class FilterOutput(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def tobytes():
        pass

    @abc.abstractmethod
    def frombytes():
        pass


class TileFilterOutput(FilterOutput):
    def __init__(self, indices=None, tiles=None):
        """Filter Output for tiles
        """
        self.indices = indices
        self.tiles = tiles
        if (indices is not None) and (tiles is not None):
            self._mappings = dict(zip(indices, tiles))

    def tobytes(self):
        return pickle.dumps(self._mappings)

    def frombytes(self, serialized):
        self._mappings = pickle.loads(serialized)
        self.indices = self._mappings.keys()
        self.tiles = self._mappings.values()
