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
    horizontal_num = math.ceil(im_w / float(w))
    vertical_num = math.ceil(im_h / float(h))
    # horizontal idx first
    slices = [[0] * vertical_num] * horizontal_num
    print('total {}x{} slices'.format(horizontal_num, vertical_num))
    for v_idx in range(1, vertical_num):
        for h_idx in range(1, horizontal_num):
            print('slicing ({}, {})'.format(h_idx, v_idx))
            slice_x = (h_idx - 1) * w
            slice_y = (v_idx - 1) * h
            slice_w = min(w, im_w - slice_x)
            slice_h = min(h, im_h - slice_y)
            current_slice = im[slice_y: slice_y + slice_h,
                            slice_x: slice_x + slice_w]
            slices[h_idx][v_idx] = current_slice
    return slices


def slice_images(input_dir, output_dir, slice_w, slice_h):
    file_paths = glob.glob(os.path.join(input_dir, '*'))
    for file_path in file_paths:
        print("slicing file {}".format(file_path))
        output_prefix, ext = os.path.splitext(os.path.basename(file_path))
        im = cv2.imread(file_path)
        slices = slice_image(im, slice_w, slice_h)
        for h_idx in len(slices):
            for v_idx in len(slices[h_idx]):
                output_name = output_prefix + "_{}_{}".format(h_idx, v_idx) + ext
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, slices[h_idx][v_idx])


def munich_split_images():
    pass


if __name__ == "__main__":
    fire.Fire()
