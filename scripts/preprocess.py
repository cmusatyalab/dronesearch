#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import shutil

import cv2
import fire
import glob
import os
import math

import io_util


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
            h_idx_format_string = io_util.get_prefix0_format_string(len(slices))
            v_idx_format_string = io_util.get_prefix0_format_string(len(slices[h_idx]))
            idx_format_string = h_idx_format_string + "_" + v_idx_format_string
            for v_idx in range(len(slices[h_idx])):
                output_name = output_prefix + "_" + idx_format_string.format(h_idx, v_idx) + ext
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, slices[h_idx][v_idx])


def get_grid_shape(file_path_pattern):
    """Return how many small pictures are cropped from the base image horizontally and vertically"""
    slice_file_paths = glob.glob(file_path_pattern)
    assert slice_file_paths
    slice_file_basenames = [os.path.splitext(os.path.basename(file_path))[0] for file_path in slice_file_paths]
    vertical_indices = [int(basename.split("_")[-1]) for basename in slice_file_basenames]
    horizontal_indices = [int(basename.split("_")[-2]) for basename in slice_file_basenames]
    return max(vertical_indices) + 1, max(horizontal_indices) + 1


def get_slice_shape(file_path_pattern):
    """Return the resolution of each slice (small picture)."""
    slice_file_paths = glob.glob(file_path_pattern)
    assert slice_file_paths
    sample_file = sorted(slice_file_paths)[0]
    im = cv2.imread(sample_file)
    return im.shape[0], im.shape[1]


def group_sliced_images_by_label(image_dir, annotation_dir, output_dir):
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
        slice_contain_roi_bitmap = [[False for _ in range(grid_h)] for _ in range(grid_w)]

        image_annotations = annotations[annotations['imageid'] == imageid]
        for _, image_annotation in image_annotations.iterrows():
            # which cell does this annotation fall into
            xmin, ymin, xmax, ymax = image_annotation["xmin"], image_annotation["ymin"], image_annotation["xmax"], \
                                     image_annotation["ymax"]
            # upper left, upper right, lower left, lower right
            key_points = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
            for (x, y) in key_points:
                # due to rectifying bounding boxes to be rectangles,
                # annotations have points that are beyond boundry of image resolutions
                grid_x, grid_y = int(x / slice_w), int(y / slice_h)
                grid_x = min(grid_w-1, max(grid_x, 0))
                grid_y = min(grid_h-1, max(grid_y, 0))
                if imageid == '4K0G0010' and grid_x == 0 and grid_y == 3:
                    import pdb
                    pdb.set_trace()
                print("marking grid cell ({}, {}) as positive".format(grid_x, grid_y))
                slice_contain_roi_bitmap[grid_x][grid_y] = True

        # move slices based on bitmap
        for h_idx in range(len(slice_contain_roi_bitmap)):
            for v_idx in range(len(slice_contain_roi_bitmap[h_idx])):
                h_idx_format_string = io_util.get_prefix0_format_string(grid_w)
                v_idx_format_string = io_util.get_prefix0_format_string(grid_h)
                idx_format_string = h_idx_format_string + "_" + v_idx_format_string
                idx_string = idx_format_string.format(h_idx, v_idx)
                image_file_paths = glob.glob(os.path.join(image_dir, "*{}*{}*").format(imageid, idx_string))
                assert len(image_file_paths) == 1

                image_file_path = image_file_paths[0]
                image_output_dir = group_dir_paths[slice_contain_roi_bitmap[h_idx][v_idx]]
                shutil.copyfile(image_file_path, os.path.join(image_output_dir, os.path.basename(image_file_path)))


if __name__ == "__main__":
    fire.Fire()
