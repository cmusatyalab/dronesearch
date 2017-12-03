#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cPickle as pickle
import collections
import cv2
import glob
import os
import random
import math
import numpy as np
import shutil
import sys

import annotation
import annotation_stats

import fire
import logzero
from logzero import logger

import io_util
import preprocess

sys.path.append('mobilenet/research/slim')
import datasets.convert_twoclass_tile

logzero.logfile("setup_train.log", maxBytes=1e6, backupCount=3, mode='w')

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


_populate_datasets_info()


def setup_train_dir_with_extra_negative():
    for dataset_name, dataset_info in dataset.iteritems():
        output_path = dataset_info['experiment_dir']
        logger.debug('setting up training directory for {} at {}'.format(
            dataset_name, output_path))
        io_util.create_dir_if_not_exist(output_path)
        for split in ['train', 'test']:
            output_file_path = os.path.join(output_path,
                                            '{}.pkl'.format(split))
            preprocess.sample_train_test_frames_with_extra_negative(
                dataset_name,
                dataset_info['tile_classification_annotation_dir'],
                dataset_info['sample_num_per_video'], dataset_info[split],
                dataset_info['extra_negative_dataset'],
                dataset_info['extra_negative_annotation_dirs'],
                dataset_info['extra_negative_sample_num_per_video'],
                dataset_info['extra_negative_video_ids'], output_file_path)

        # create photo dir links to images
        io_util.create_dir_if_not_exist(os.path.join(output_path, 'photos'))
        for dataset_name, dataset_info in dataset.iteritems():
            link_path = os.path.join(output_path, 'photos', dataset_name)
            if not os.path.exists(link_path):
                os.symlink(
                    os.path.abspath(dataset_info['image_dir']),
                    link_path)
            else:
                logger.info('{} exists. skip linking'.format(link_path))


def setup_tfrecords():
    for dataset_name, dataset_info in dataset.iteritems():
        logger.debug('preparing tf record for {}'.format(dataset_name))
        datasets.convert_twoclass_tile.run(
            dataset_dir=dataset_info['experiment_dir'],
            mode='train',
            tile_width=224,
            tile_height=224)


if __name__ == "__main__":
    fire.Fire()
