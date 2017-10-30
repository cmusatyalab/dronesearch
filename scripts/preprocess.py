#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import cv2
import fire
import glob
import math
import numpy as np
import os
import random
import shutil

import io_util
import pandas as pd


def slice_image(im, w, h):
    """Slice image into tiles of width w and height h.

    The returned slice order is horizontal-first.
    """
    im_h, im_w, _ = im.shape
    horizontal_num = int(math.ceil(im_w / float(w)))
    vertical_num = int(math.ceil(im_h / float(h)))

    # horizontal idx first, top left is (0,0)
    slices = []
    print('total {}x{} slices'.format(horizontal_num, vertical_num))
    for h_idx in range(0, horizontal_num):
        vertical_slice = []
        for v_idx in range(0, vertical_num):
            print('slicing ({}, {})'.format(h_idx, v_idx))
            slice_x = h_idx * w
            slice_y = v_idx * h
            slice_w = min(w, im_w - slice_x)
            slice_h = min(h, im_h - slice_y)
            current_slice = im[slice_y: slice_y + slice_h,
                               slice_x: slice_x + slice_w]
            vertical_slice.append(current_slice)
        slices.append(vertical_slice)
    return slices


def slice_images(input_dir, output_dir, slice_w=225, slice_h=225):
    """"Used by Munich dataset to slice images into smaller tiles."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = glob.glob(os.path.join(input_dir, '*'))
    for file_path in file_paths:
        print("slicing file {}".format(file_path))
        output_prefix, ext = os.path.splitext(os.path.basename(file_path))
        im = cv2.imread(file_path)
        slices = slice_image(im, slice_w, slice_h)
        for h_idx in range(len(slices)):
            h_idx_format_string = io_util.get_prefix0_format_string(
                len(slices))
            v_idx_format_string = io_util.get_prefix0_format_string(
                len(slices[h_idx]))
            idx_format_string = h_idx_format_string + "_" + v_idx_format_string
            for v_idx in range(len(slices[h_idx])):
                output_name = output_prefix + "_" + \
                    idx_format_string.format(h_idx, v_idx) + ext
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, slices[h_idx][v_idx])


def get_grid_shape(file_path_pattern):
    """Return how many small pictures are cropped from the base image horizontally
    and vertically."""
    slice_file_paths = glob.glob(file_path_pattern)
    assert slice_file_paths
    slice_file_basenames = [os.path.splitext(os.path.basename(file_path))[
        0] for file_path in slice_file_paths]
    vertical_indices = [int(basename.split("_")[-1])
                        for basename in slice_file_basenames]
    horizontal_indices = [int(basename.split("_")[-2])
                          for basename in slice_file_basenames]
    return max(vertical_indices) + 1, max(horizontal_indices) + 1


def get_slice_shape(file_path_pattern):
    """Return the resolution of each slice (small picture)."""
    slice_file_paths = glob.glob(file_path_pattern)
    assert slice_file_paths
    sample_file = sorted(slice_file_paths)[0]
    im = cv2.imread(sample_file)
    return im.shape[0], im.shape[1]


def group_sliced_images_by_label(image_dir, annotation_dir, output_dir):
    """Group munich images into postivie and negative.

    If the cropped image has a car, then it's positive, otherwise negative.
    """
    annotations = io_util.load_munich_annotation(annotation_dir)
    imageids = set(annotations['imageid'].tolist())

    group_dir_paths = {True: os.path.join(output_dir, "positive"),
                       False: os.path.join(output_dir, "negative")}
    for group_dir_path in group_dir_paths.values():
        if not os.path.exists(group_dir_path):
            os.makedirs(group_dir_path)

    for imageid in imageids:
        print("processing slices from base image id: {}".format(imageid))
        file_path_pattern = os.path.join(image_dir, "*{}*".format(imageid))
        grid_h, grid_w = get_grid_shape(file_path_pattern)
        print("grid size: ({}, {})".format(grid_w, grid_h))
        slice_h, slice_w = get_slice_shape(file_path_pattern)
        slice_contain_roi_bitmap = [
            [False for _ in range(grid_h)] for _ in range(grid_w)]

        image_annotations = annotations[annotations['imageid'] == imageid]
        for _, image_annotation in image_annotations.iterrows():
            # which cell does this annotation fall into
            xmin, ymin, xmax, ymax = image_annotation["xmin"], \
                image_annotation["ymin"], \
                image_annotation["xmax"], \
                image_annotation["ymax"]
            # upper left, upper right, lower left, lower right
            key_points = [(xmin, ymin), (xmax, ymin),
                          (xmin, ymax), (xmax, ymax)]
            for (x, y) in key_points:
                # due to rectifying bounding boxes to be rectangles,
                # annotations have points that are beyond boundry of image
                # resolutions
                grid_x, grid_y = int(x / slice_w), int(y / slice_h)
                grid_x = min(grid_w - 1, max(grid_x, 0))
                grid_y = min(grid_h - 1, max(grid_y, 0))
                print("marking grid cell ({}, {}) as positive".format(grid_x,
                                                                      grid_y))
                slice_contain_roi_bitmap[grid_x][grid_y] = True

        # move slices based on bitmap
        for h_idx in range(len(slice_contain_roi_bitmap)):
            for v_idx in range(len(slice_contain_roi_bitmap[h_idx])):
                h_idx_format_string = io_util.get_prefix0_format_string(grid_w)
                v_idx_format_string = io_util.get_prefix0_format_string(grid_h)
                idx_format_string = h_idx_format_string + \
                    "_" + v_idx_format_string
                idx_string = idx_format_string.format(h_idx, v_idx)
                image_file_paths = glob.glob(os.path.join(
                    image_dir, "*{}*{}*").format(imageid, idx_string))
                assert len(image_file_paths) == 1

                image_file_path = image_file_paths[0]
                image_output_dir = group_dir_paths[
                    slice_contain_roi_bitmap[h_idx][v_idx]]
                shutil.copyfile(image_file_path, os.path.join(
                    image_output_dir, os.path.basename(image_file_path)))


def flatten_stanford_dir(input_dir, output_dir):
    io_util.flatten_directory_with_symlink(input_dir, output_dir)


def get_positive_annotation_mask(annotations, videoids):
    lost_mask = (annotations['lost'] == 0)
    label_mask = annotations['label'].isin(['Car', 'Bus'])
    mask = label_mask & lost_mask
    return mask


def group_stanford_images(image_dir,
                          annotation_dir,
                          output_dir,
                          max_positive_num=10000,
                          max_negative_num=20000):
    """Group stanford images into postivie and negative.

    If the image has a car/bus, then it's positive, otherwise negative.

    image_dir: should be the either the "train" or "test" dir with first level
    being video names and second level images.

    """
    VIDEO_DIR_SUFFIX = '_video.mov'
    IMAGE_EXT = '.jpg'

    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    print('loaded annotations.')
    video_list = os.listdir(image_dir)
    videoids = [videoname.replace('_video.mov', '') for videoname in
                video_list]
    # restrict to videoids in image_dir only
    annotations = annotations[annotations['videoid'].isin(videoids)]
    image_dir_abspath = os.path.abspath(image_dir)
    sample_num_lut = {
        'positive': max_positive_num,
        'negative': max_negative_num
    }

    for category in ['positive', 'negative']:
        print('gathering images for category {}'.format(category))
        category_output_dir = os.path.join(output_dir, category)
        io_util.create_dir_if_not_exist(category_output_dir)
        mask = get_positive_annotation_mask(annotations, videoids)
        target_annotations = annotations[mask].copy()
        # groupby videoid and frameid to remove duplicates
        unique_target_annotations = target_annotations.groupby(
            ['videoid', 'frameid']).sum()
        # list of (video ids, frame ids)
        unique_target_tuples = unique_target_annotations.index.values
        if category == 'negative':
            # when negative, we override the unique_target_tuples with the
            # negation of target tuples
            unique_annotations = annotations.groupby(
                ['videoid', 'frameid']).sum()
            unique_tuples_set = set(unique_annotations.index.values)
            unique_target_tuples_set = set(unique_target_tuples)
            negative_set = unique_tuples_set - unique_target_tuples_set
            unique_target_tuples = np.asarray(list(negative_set))

        qualified_num = len(unique_target_tuples)
        print('category {} has {} qualified frames'.format(
            category, qualified_num))
        tuples_indices = range(qualified_num)
        random.shuffle(tuples_indices)
        sample_num = sample_num_lut[category]
        print('sampling {} frames'.format(sample_num))
        sample_indices = tuples_indices[:sample_num]
        samples = unique_target_tuples[sample_indices]
        # extracted frame images starts from 1
        # frameid in the dataset annotations starts from 0
        src_image_list = map(lambda (videoid, frameid):
                             os.path.join(image_dir_abspath,
                                          videoid +
                                          VIDEO_DIR_SUFFIX,
                                          '{:010d}'.format(int(frameid)+1) +
                                          IMAGE_EXT), samples)

        # create symlink to source images in output_dir
        for src_image_path in src_image_list:
            relpath = os.path.relpath(src_image_path,
                                      image_dir)
            dst_relpath = relpath.replace(os.path.sep, '_')
            dst_image_path = os.path.join(category_output_dir,
                                          dst_relpath)
            os.symlink(src_image_path, dst_image_path)


def split_train_test(data_list_file, output_dir, test_percentage=0.1):
    """Split train test set.

    data_list_file: a file with each data occupying a line
    output_dir: output_dir that contains train.txt and test.txt
    """
    with open(data_list_file, 'r') as f:
        data_list = f.read().splitlines()
    # use seed so that we can get consistent results
    random.seed('hash')
    random.shuffle(data_list)
    num_test = int(len(data_list) * test_percentage)
    test_list = data_list[:num_test]
    train_list = data_list[num_test:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    io_util.write_list_to_file(train_list,
                               os.path.join(output_dir, 'train.txt'))
    io_util.write_list_to_file(test_list,
                               os.path.join(output_dir, 'test.txt'))


if __name__ == "__main__":
    fire.Fire()
