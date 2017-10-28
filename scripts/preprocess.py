#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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
    for v_idx in range(0, vertical_num):
        horizontal_slices = []
        for h_idx in range(0, horizontal_num):
            print('slicing ({}, {})'.format(h_idx, v_idx))
            slice_x = h_idx * w
            slice_y = v_idx * h
            slice_w = min(w, im_w - slice_x)
            slice_h = min(h, im_h - slice_y)
            current_slice = im[slice_y: slice_y + slice_h,
                            slice_x: slice_x + slice_w]
            horizontal_slices.append(current_slice)
        slices.append(horizontal_slices)
    return slices


def slice_images(input_dir, output_dir, slice_w, slice_h):
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


if __name__ == "__main__":
    fire.Fire()
