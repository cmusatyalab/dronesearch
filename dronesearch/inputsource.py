
# Drone Search
#
# A computer vision pipeline for live video search on drone video
# feeds leveraging edge servers.
#
# Copyright (C) 2018-2019 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input Sources for Retrieving Video Frames.
"""

import abc

import cv2
from logzero import logger


class InputSource(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read():
        pass

    @abc.abstractmethod
    def close():
        pass


class OpenCVInputSource(InputSource):
    def __init__(self, source):
        """Video Input Source using OpenCV. Return RGB image.

        """
        self.source = source

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, True)

    def read(self):
        ret, frame = self._cap.read()
        if not ret:
            logger.error('Failed to retrieve a frame from {}'.format(
                self.source))
            return None
        return frame

    def close(self):
        self._cap.release()
