"""Filters for on-board processing
"""

import abc
import math
import pickle
import time

import cv2
import numpy as np
from logzero import logger

import tensorflow as tf
from dronesearch import utils


class DroneFilter(object, metaclass=abc.ABCMeta):
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
                 input_std, input_layer, output_layer, positive_label):
        self.name = name
        self.model_file = model_file
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.input_mean = int(input_mean)
        self.input_std = int(input_std)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self._class = int(positive_label)
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

        self._sess = tf.Session(graph=self._graph)

    @utils.timeit
    def process(self, image):
        tiles = [image]
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
        logger.debug('score for class {0} is: {1}'.format(self._class,
                                                          predictions[0][self._class]))
        result = None

        if (predictions[0][self._class] > 0.6):
            result = ImageFilterOutput(image)

        return result

    def close(self):
        self._sess.close()

    def update():
        raise NotImplementedError()


class FilterOutput(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def tobytes():
        pass

    @abc.abstractmethod
    def frombytes():
        pass


class ImageFilterOutput(FilterOutput):
    def __init__(self, image=None):
        self.image = image

    def tobytes(self):
        return pickle.dumps(self.image)

    def frombytes(self, serialized):
        self.image = pickle.loads(serialized)
