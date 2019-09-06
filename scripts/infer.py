#!/usr/bin/env python

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

"""Script for running inference using trained models.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cPickle as pickle
import cv2
import glob
import math
import os
import collections
import copy
import numpy as np
import shutil
import sys
import random

import annotation
import annotation_stats
import fire
import logzero
import preprocess
import io_util
from logzero import logger

sys.path.append('mobilenet/research/slim')
import infer_tile_classifier as infer_tile

logzero.logfile("infer.log", maxBytes=1e6, backupCount=3, mode='w+')

########################################
# populate dir specific dataset info
########################################
dataset = annotation_stats.dataset


def _populate_datasets_info():
    for dataset_name, dataset_info in dataset.iteritems():
        dataset_info['tile_classification_annotation_dir'] = os.path.join(
            dataset_name, 'classification_448_224_224_224_annotations')
        dataset_info['sample_num_per_video'] = 2000
        dataset_info['extra_negative_annotation_dirs'] = []
        dataset_info['extra_negative_video_ids'] = []
        for extra_negative_dataset_name in dataset_info[
                'extra_negative_dataset']:
            dataset_info['extra_negative_annotation_dirs'].append(
                os.path.join(extra_negative_dataset_name,
                             'classification_448_224_224_224_annotations'))
            dataset_info['extra_negative_video_ids'].append(
                dataset[extra_negative_dataset_name]['train'])
        dataset_info['extra_negative_sample_num_per_video'] = 2000
        dataset_info['image_dir'] = os.path.join(dataset_name,
                                                 'images_448_224')
        dataset_info['experiment_dir'] = os.path.join(
            dataset_name, 'experiments',
            'classification_448_224_224_224_extra_negative')
        dataset_info['checkpoint_dir'] = os.path.join(
            dataset_info['experiment_dir'], 'logs_all_layers_40000')
        dataset_info['test_inference_dir'] = os.path.join(
            dataset_info['experiment_dir'], 'test_inference')
        io_util.create_dir_if_not_exist(dataset_info['test_inference_dir'])


_populate_datasets_info()


def _get_infer_tile_setup(dataset_name, dataset_info):
    infer_experiments = []
    # stanford has videos that h > w
    if dataset_name == 'stanford':
        # horizontal videos
        flags = []
        flags.append(dataset_info['checkpoint_dir'])
        flags.append(dataset_info['image_dir'])
        flags.append('Predictions,AvgPool_1a')
        flags.append(
            os.path.join(dataset_info['test_inference_dir'],
                         '{}_test_inference_results_horizontal.pkl'.format(
                             dataset_name)))
        flags.append(2)
        flags.append(1)
        flags.append(448)
        flags.append(224)
        flags.append(100)
        flags.append(1.0)
        flags.append(','.join(dataset_info['test_horizontal']))
        flags.append(dataset_name + '/')
        assert len(flags) == 12
        infer_experiments.append(flags)
        # vertical videos
        flags = []
        flags.append(dataset_info['checkpoint_dir'])
        flags.append(dataset_info['image_dir'])
        flags.append('Predictions,AvgPool_1a')
        flags.append(
            os.path.join(
                dataset_info['test_inference_dir'],
                '{}_test_inference_results_vertical.pkl'.format(dataset_name)))
        flags.append(1)
        flags.append(2)
        flags.append(224)
        flags.append(448)
        flags.append(100)
        flags.append(1.0)
        flags.append(','.join(dataset_info['test_vertical']))
        flags.append(dataset_name + '/')
        assert len(flags) == 12
        infer_experiments.append(flags)
    else:
        flags = []
        flags.append(dataset_info['checkpoint_dir'])
        flags.append(dataset_info['image_dir'])
        flags.append('Predictions,AvgPool_1a')
        flags.append(
            os.path.join(dataset_info['test_inference_dir'],
                         '{}_test_inference_results.pkl'.format(dataset_name)))
        flags.append(2)
        flags.append(1)
        flags.append(448)
        flags.append(224)
        flags.append(100)
        flags.append(1.0)
        flags.append(','.join(dataset_info['test']))
        flags.append(dataset_name + '/')
        assert len(flags) == 12
        infer_experiments.append(flags)
    return infer_experiments


def _run_infer_experiment(flags):
    logger.info('running inference with flags: {}'.format(flags))
    infer_tile.FLAGS.checkpoint_path = flags[0]
    infer_tile.FLAGS.input_dir = flags[1]
    infer_tile.FLAGS.output_endpoint_names = flags[2]
    infer_tile.FLAGS.result_file = flags[3]
    infer_tile.FLAGS.grid_w = flags[4]
    infer_tile.FLAGS.grid_h = flags[5]
    infer_tile.FLAGS.image_w = flags[6]
    infer_tile.FLAGS.image_h = flags[7]
    infer_tile.FLAGS.batch_size = flags[8]
    infer_tile.FLAGS.max_gpu_memory_fraction = flags[9]
    infer_tile.FLAGS.video_ids = flags[10]
    infer_tile.FLAGS.tile_id_prefix = flags[11]
    infer_tile.main('unused')


def infer():
    for dataset_name, dataset_info in dataset.iteritems():
        logger.info('inferring: {}'.format(dataset_name))
        infer_experiments = _get_infer_tile_setup(dataset_name, dataset_info)
        for flags in infer_experiments:
            _run_infer_experiment(flags)
        # todo : infer on extra negative sets as well
        for negative_dataset_name in dataset_info['extra_negative_dataset']:
            logger.info(
                'inferring negative dataset: {}'.format(negative_dataset_name))
            dataset_info_for_negative = copy.deepcopy(dataset_info)
            dataset_info_for_negative['image_dir'] = dataset[
                negative_dataset_name]['image_dir']
            dataset_info_for_negative['test'] = dataset[negative_dataset_name][
                'test']
            if 'test_vertical' in dataset[negative_dataset_name]:
                dataset_info_for_negative['test_vertical'] = dataset[
                    negative_dataset_name]['test_vertical']
                dataset_info_for_negative['test_horizontal'] = dataset[
                    negative_dataset_name]['test_horizontal']

            infer_negative_experiments = _get_infer_tile_setup(
                negative_dataset_name, dataset_info_for_negative)
            for flags in infer_negative_experiments:
                _run_infer_experiment(flags)


if __name__ == "__main__":
    fire.Fire()
