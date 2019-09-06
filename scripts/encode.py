
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

"""Analyze stanford results."""
from __future__ import absolute_import, division, print_function

import cPickle as pickle
import cv2
import glob
import json
import collections
import math
import os
import numpy as np
import subprocess

import fire
import annotation_stats
import matplotlib
import io_util
import redis

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from itertools import izip_longest

import logzero
from logzero import logger
logzero.logfile("encode.log", maxBytes=1e6, backupCount=3, mode='a')


def get_all_video_iter(dataset_ids=[]):
    for dataset_name in annotation_stats.dataset:
        if dataset_ids:
            if dataset_name not in dataset_ids:
                continue
        for video_id in annotation_stats.dataset[dataset_name]['test']:
            yield ('{}_{}'.format(dataset_name, video_id),
                   os.path.join(dataset_name, 'images', video_id))


def encode_datasets(dataset_ids, output_dir, crf=23):
    video_iter = get_all_video_iter(dataset_ids)
    procs = []
    io_util.create_dir_if_not_exist(output_dir)
    for dataset_video_id, frame_sequence_dir in video_iter:
        output_file_path = os.path.join(output_dir, dataset_video_id + '.mp4')
        procs.append(
            encode_images_to_h264(
                frame_sequence_dir, output_file_path, crf=crf))

    for proc in procs:
        stdout_value, stderr_value = proc.communicate()
        ret_val = proc.returncode
        logger.debug('{}: returns {}\n{}\n{}'.format(
            proc, ret_val, stdout_value, stderr_value))


def get_jpeg_size_info(dataset_ids, unit='KB'):
    unit_to_power = {
        'KB': 1,
        'MB': 2,
    }
    assert unit in unit_to_power
    divider = 1024**unit_to_power[unit]
    logger.debug('unit: {}'.format(unit))
    video_iter = get_all_video_iter(dataset_ids)
    all_image_sizes = []
    for dataset_video_id, frame_sequence_dir in video_iter:
        jpeg_image_sizes = [
            os.path.getsize(os.path.join(frame_sequence_dir, f))
            for f in os.listdir(frame_sequence_dir)
        ]
        all_image_sizes.extend(jpeg_image_sizes)
        logger.debug(
            '{} has {} images, totaling {:.2f}, avg {:.2f} , std {:.2f} '.
            format(dataset_video_id, len(jpeg_image_sizes),
                   np.sum(jpeg_image_sizes) / divider,
                   np.average(jpeg_image_sizes) / divider,
                   np.std(jpeg_image_sizes) / divider))
    logger.debug(
        'In total {} images, totaling {:.2f} , avg {:.2f} , std {:.2f} '.
        format(
            len(all_image_sizes),
            np.sum(all_image_sizes) / divider,
            np.average(all_image_sizes) / divider,
            np.std(all_image_sizes) / divider))

    #     jpeg_images_num.append(len(jpeg_images))
    #     frame_sequence_sizes.append(frame_sequence_size)
    # jpeg_images_num = np.array(jpeg_images_num)
    # frame_sequence_sizes = np.array(frame_sequence_sizes)
    # logger.debug(
    #     'total size: {}, total number of jpeg files: {}, average: {}, std: {}'.
    #     format(np.sum(frame_sequence_sizes), np.sum(jpeg_images_num), ))


def encode_video_h264(video_path, output_file_path, crf):
    args = [
        'ffmpeg', '-i',
        video_path, '-vcodec', 'libx264',
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-crf',
        str(crf), output_file_path
    ]
    logger.debug(' '.join(args))
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def encode_images_to_h264(frame_sequence_dir, output_file_path, crf):
    args = [
        'ffmpeg', '-f', 'image2', '-framerate', '30', '-i',
        os.path.join(frame_sequence_dir, '%010d.jpg'), '-vcodec', 'libx264',
        '-vf', "scale=trunc(iw/2)*2:trunc(ih/2)*2", '-crf',
        str(crf), output_file_path
    ]
    logger.debug(' '.join(args))
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


if __name__ == '__main__':
    fire.Fire()
