#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cPickle
import cv2
import fire
import glob
import os

import numpy as np

import io_util

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw_annotations_on_image(input_image_path, annotations,
                              output_image_path):
    """Draw annotations on images.
    Annotations should be a panda dataframe
    with xmin, ymin, xmax, ymax, lost columns"""
    bgr_im = cv2.imread(input_image_path)
    for _, annotation in annotations.iterrows():
        if 'lost' in annotation and annotation['lost']:
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


def visualize_annotations_in_image(input_image_path, annotation_file_path,
                                   annotation_format, output_image_path):
    """Visualize annotation in an image."""
    supported_parse_function = {
        "vatic": io_util.parse_vatic_annotation_file,
        "munich": io_util.parse_munich_annotation_file
    }
    if annotation_format not in supported_parse_function.keys():
        raise ValueError(
            "Annotation format Not Supported. Currently supports {}, not {}".
            format(supported_parse_function.keys(), annotation_format))
    image_annotations = supported_parse_function[annotation_format](
        annotation_file_path)
    draw_annotations_on_image(input_image_path, image_annotations,
                              output_image_path)


def visualize_annotations_in_frame_sequence(frame_sequence_dir,
                                            annotation_dir,
                                            output_dir,
                                            labels,
                                            video_id=""):
    frame_file_list = sorted(glob.glob(os.path.join(frame_sequence_dir, '*')))
    annotations = io_util.load_annotation_from_dir(
        annotation_dir, io_util.parse_vatic_annotation_file)
    io_util.create_dir_if_not_exist(output_dir)
    if video_id:
        print("filtering through video_id: {}".format(video_id))
        annotations = annotations[annotations['videoid'] == video_id]
    labels = labels.split(',')
    lost_mask = (annotations['lost'] == 0)
    label_mask = annotations['label'].isin(labels)
    mask = label_mask & lost_mask
    annotations = annotations[mask]
    assert len(annotations) > 0

    for frame_file in frame_file_list:
        frame_base_file = os.path.basename(frame_file)
        (frame_seq, ext) = os.path.splitext(frame_base_file)
        frameid = int(frame_seq)

        print("processing: %s" % frameid)
        frame_annotations = annotations[annotations['frameid'] == frameid]

        output_frame_path = os.path.join(output_dir, frame_base_file)
        draw_annotations_on_image(frame_file, frame_annotations,
                                  output_frame_path)


def _get_keys_by_id_prefix(my_dict, key_prefix):
    key_prefix += '_'
    keys = [key for key in my_dict.iterkeys() if key.startswith(key_prefix)]
    return keys


def _get_tile_size_from_ratio(im, long_edge_ratio, short_edge_ratio):
    im_h, im_w, _ = im.shape
    if im_h > im_w:
        tile_height = int(im_h * long_edge_ratio)
        tile_width = int(im_w * short_edge_ratio)
    else:
        tile_width = int(im_w * long_edge_ratio)
        tile_height = int(im_h * short_edge_ratio)
    return tile_height, tile_width


def visualize_predictions_in_frame_sequence(frame_sequence_dir,
                                            result_dir,
                                            output_dir,
                                            video_id="",
                                            prediction_threshold=0.5,
                                            long_edge_ratio=0.5,
                                            short_edge_ratio=1):
    frame_file_list = sorted(glob.glob(os.path.join(frame_sequence_dir, '*')))
    predictions = io_util.load_all_pickles_from_dir(result_dir)

    io_util.create_dir_if_not_exist(output_dir)
    if video_id:
        print("filtering through video_id: {}".format(video_id))
        video_predictions_ids = _get_keys_by_id_prefix(predictions, video_id)
        predictions = {
            k: v
            for k, v in predictions.items() if k in video_predictions_ids
        }
    for frame_file in frame_file_list:
        frame_base_file = os.path.basename(frame_file)
        (frame_seq, ext) = os.path.splitext(frame_base_file)
        frame_id = int(frame_seq)

        print("processing: %s" % frame_id)
        prediction_image_id = '{}_{}'.format(video_id, frame_id)
        prediction_tile_ids = _get_keys_by_id_prefix(predictions,
                                                     prediction_image_id)
        im = cv2.imread(frame_file)
        orig_im = np.copy(im)
        output_frame_path = os.path.join(output_dir, frame_base_file)
        tile_height, tile_width = _get_tile_size_from_ratio(
            im, long_edge_ratio, short_edge_ratio)
        for prediction_tile_id in prediction_tile_ids:
            if predictions[prediction_tile_id][1] < prediction_threshold:
                grid_x, grid_y = int(prediction_tile_id.split('_')[-2]), int(
                    prediction_tile_id.split('_')[-1])
                tile_x = grid_x * tile_width
                tile_y = grid_y * tile_height
                # tile_to_grey = np.copy(im[tile_y:tile_y + tile_height, tile_x:
                #                           tile_x + tile_width])
                # tile_to_grey = cv2.cvtColor(tile_to_grey, cv2.COLOR_BGR2GRAY)
                # # make it back to 3 channels so that
                # # we can put it in the original image
                # tile_to_grey = cv2.cvtColor(tile_to_grey, cv2.COLOR_GRAY2BGR)
                # im[tile_y:tile_y + tile_height, tile_x:
                #    tile_x + tile_width] = tile_to_grey
                im[tile_y:tile_y + tile_height, tile_x:
                   tile_x + tile_width] = 0
        combined_im = np.concatenate((orig_im, im), axis=0)
        cv2.imwrite(output_frame_path, combined_im)


def rename_frame_sequence_for_avconv(frame_sequence_dir, output_dir):
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
    frame_file_list = sorted(
        glob.glob(os.path.join(os.path.abspath(frame_sequence_dir), '*')))
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
        dst_file = os.path.join(output_dir,
                                output_file_name_pat.format(index + 1))
        os.symlink(frame_file, dst_file)


def vis_prc(precision_recall_file_path, output_file_path):
    """Plot precision and recall curve."""
    with open(precision_recall_file_path, 'r') as f:
        data = cPickle.load(f)
        recall = data['rec']
        precision = data['prec']
        average_precision = data['ap']
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: Average Precision={0:0.2f}'.format(
            average_precision))
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close()


if __name__ == "__main__":
    fire.Fire()
