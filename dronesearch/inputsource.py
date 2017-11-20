#!/usr/bin/env python
"""Filters for on-board processing
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc

import cv2
from logzero import logger


class InputSource(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def read():
        pass

    @abc.abstractmethod
    def close():
        pass


class OpenCVInputSource(InputSource):
    def __init__(self, source):
        """Video Input Source using OpenCV. Return RGB image

        """
        self.source = source

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB, True)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            logger.error('Failed to retrieve a frame from {}'.format(
                self.source))
            return None
        return frame

    def close(self):
        self._cap.release()
