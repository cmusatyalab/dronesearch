#!/usr/bin/env python

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *

import cv2
import fire
import glob
import os

import io_util


def draw_annotations_on_image(input_image_path,
                              annotations,
                              output_image_path):
    """Draw annotations on images.
    Annotations should be a panda dataframe
    with xmin, ymin, xmax, ymax, lost columns"""
    bgr_im = cv2.imread(input_image_path)
    for _, annotation in annotations.iterrows():
        if annotation['lost']:
            continue
        xmin = annotation['xmin']
        ymin = annotation['ymin']
        xmax = annotation['xmax']
        ymax = annotation['ymax']
        color = (255, 0, 0)
        thickness = 3
        cv2.rectangle(bgr_im, (xmin, ymin), (xmax, ymax), color, thickness)
    if output_image_path:
        cv2.imwrite(output_image_path, bgr_im)
    return bgr_im


def visualize_annotations_in_frame_sequence(frame_sequence_dir,
                                            annotation_file_path,
                                            output_dir):
    frame_file_list = sorted(glob.glob(os.path.join(frame_sequence_dir, '*')))
    annotations = io_util.parse_annotation_file(annotation_file_path)
    os.makedirs(output_dir)
    for frame_file in frame_file_list:
        frame_base_file = os.path.basename(frame_file)
        (frame_seq, ext) = os.path.splitext(frame_base_file)
        frameid = int(frame_seq)

        print("processing: %s" % frameid)
        frame_annotations = annotations[annotations['frameid'] == frameid]

        output_frame_path = os.path.join(output_dir, frame_base_file)
        draw_annotations_on_image(frame_file,
                                  frame_annotations,
                                  output_frame_path)


def rename_frame_sequence_for_avconv(frame_sequence_dir,
                                     output_dir):
    """Rename image files so that avconv can combine them into a video.

    Avconv cmd: avconv -r 30 -i /frames/%3d.jpg output_video.mov

    Above cmd assumes image file names in /frames are named as consecutive
    numbers. e.g. 001.jpg, 002.jpg, 003.jpg. Avconv would stop combining frames
    when any discontinuity in the consecutive numbers appears. e.g. 001.jpg,
    003.jpg won't get combined since 002.jpg is missing.

    This function renames frame sequence files in the frame_sequence_dir to
    consecutive frames so that above avconv cmd can be used to combine them
    into a video.

    The order is the natural sequence of the original file names.
    """
    frame_file_list = sorted(glob.glob(
        os.path.join(os.path.abspath(frame_sequence_dir), '*')))
    if not frame_file_list:
        raise ValueError("No files found in {}".format(frame_sequence_dir))

    _, image_ext = os.path.splitext(frame_file_list[0])
    frame_num = len(frame_file_list)
    max_digit_num = len(str(frame_num))

    print(('Issue "avconv -r 30 -i {}/{}{} output_video.mov" '
           'to create a video.').format(
            output_dir, '%' + str(max_digit_num) + 'd', image_ext))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_name_pat = '{:0' + str(max_digit_num) + 'd}' + image_ext
    for index, frame_file in enumerate(frame_file_list):
        dst_file = os.path.join(
            output_dir, output_file_name_pat.format(index+1))
        os.symlink(frame_file, dst_file)


if __name__ == "__main__":
    fire.Fire(
        {'visualize_annotations_in_frame_sequence':
         visualize_annotations_in_frame_sequence,
         'rename_frame_sequence_for_avconv':
         rename_frame_sequence_for_avconv})
