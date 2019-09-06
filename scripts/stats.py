
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

"""Stats about vidoes/frame sequences in the datasets.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import collections
import cv2
import glob
import itertools
import json
import numpy as np
import operator
import os
import cPickle as pickle

import fire
import io_util
import redis


def frame_sequence_resolution(base_dir):
    frame_sequence_names = os.listdir(base_dir)
    frame_sequence_name_to_resolution = {}
    for frame_sequence_name in frame_sequence_names:
        sample_file = os.path.join(base_dir, frame_sequence_name,
                                   '{:010d}.jpg'.format(1))
        im_h, im_w, _ = cv2.imread(sample_file).shape
        frame_sequence_name_to_resolution[
            frame_sequence_name] = '{}x{}'.format(im_w, im_h)
    print(json.dumps(frame_sequence_name_to_resolution, indent=4))


if __name__ == '__main__':
    fire.Fire()
