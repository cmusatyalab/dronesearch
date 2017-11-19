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
        """Video Input Source

        """
        self.source = source
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB, True)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            raise ValueError('Failed to retrieve a frame from {}'.format(
                self.source))
        return frame

    def close(self):
        self._cap.release()
