#!/usr/bin/env python
"""Preprocess scripts for Faster-RCNN based object detections experiments.

This preprocessing script customizes input data format according to the
convention used by TPOD's py-faster-rcnn CNN
module. (https://github.com/junjuew/py-faster-rcnn)

"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import fire
import glob
import os

import io_util


def gen_tpod_image_label_line(annotations):
    """Generate a line in label_list from annotations in one image."""
    labels = []
    label_line = []
    for _, annotation in annotations.iterrows():
        if annotation['lost']:
            continue
        xmin = int(annotation['xmin'])
        ymin = int(annotation['ymin'])
        xmax = int(annotation['xmax'])
        ymax = int(annotation['ymax'])
        w = xmax - xmin
        h = ymax - ymin
        labels.append(','.join([str(xmin), str(ymin), str(w), str(h)]))
    if labels:
        # add this frame to training set
        label_line.append(';'.join(labels))
    return label_line


def gen_tpod_data_inputs(frame_sequence_dir, annotations):
    """Generate image_list and label_list for a frame sequence."""
    image_list, label_list = [], []
    frame_file_list = sorted(glob.glob(os.path.join(frame_sequence_dir, '*')))
    print("iterating over video {}".format(frame_sequence_dir))
    for frame_file in frame_file_list:
        frame_base_file = os.path.basename(frame_file)
        (frame_seq, ext) = os.path.splitext(frame_base_file)
        frameid = int(frame_seq)
        frame_annotations = annotations[annotations['frameid'] == frameid]
        label_line = gen_tpod_image_label_line(frame_annotations)
        assert len(label_line) < 2
        if label_line:
            image_list.append(frame_file)
            label_list.extend(label_line)
    return image_list, label_list


def gen_car_only_experiments(scene_frame_sequence_dir,
                             scene_annotation_dir,
                             train_list,
                             output_dir):
    """Generate TPOD data input files for training.

    Experiments for training car/bus detectors only.
    Car and bus are lumped together as a single class.

    Args:
        scene_frame_sequence_dir:
          Base directory path for scene in Stanford Campus dataset
          (e.g. 'nexus')
        train_list, test_list:
          list of videos for train and test. e.g ['video0', 'video1']
    """
    # image_list, label_list, label_name file
    image_list, label_list = [], []
    label_name_list = ["car"]

    for video_name in train_list:
        video_frame_sequence_dir = os.path.join(
            scene_frame_sequence_dir, video_name)
        video_annotations = io_util.parse_vatic_annotation_file(
            os.path.join(scene_annotation_dir, video_name, 'annotations.txt'))
        video_image_list, video_label_list = gen_tpod_data_inputs(
            video_frame_sequence_dir, video_annotations)
        image_list.extend(video_image_list)
        label_list.extend(video_label_list)

    # write these three lists to disk
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_list_file_path = os.path.join(output_dir, 'image_list.txt')
    output_label_list_file_path = os.path.join(output_dir, 'label_list.txt')
    output_label_name_file_path = os.path.join(output_dir, 'label_name.txt')
    io_util.write_list_to_file(image_list, output_image_list_file_path)
    io_util.write_list_to_file(label_list, output_label_list_file_path)
    io_util.write_list_to_file(label_name_list, output_label_name_file_path)


if __name__ == "__main__":
    fire.Fire(
        {'gen_car_only_experiments':
         gen_car_only_experiments})
