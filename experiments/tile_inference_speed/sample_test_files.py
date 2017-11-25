#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import glob
import os
import random
import shutil

import fire


OKUTAMA_TRAIN_VIDEOS = [
    '1.1.3', '1.1.2', '1.1.5', '1.1.4', '2.2.7', '2.1.7', '2.1.4', '2.2.11',
    '1.1.10'
]
OKUTAMA_TEST_VIDEOS = ['2.2.2', '2.2.4', '1.1.7']


def sample(image_dir, output_dir, sample_num=100):
    file_list = []
    for video_id in OKUTAMA_TEST_VIDEOS:
        video_file_list = glob.glob(os.path.join(image_dir, video_id, '*'))
        file_list.extend(video_file_list)
    random.shuffle(file_list)
    sample_list = file_list[:sample_num]
    for file_path in sample_list:
        relpath = os.path.relpath(file_path, image_dir)
        dst_relpath = relpath.replace(os.path.sep, '_')
        dst_file_path = os.path.join(output_dir, dst_relpath)
        shutil.copyfile(file_path, dst_file_path)


if __name__ == "__main__":
    fire.Fire()
