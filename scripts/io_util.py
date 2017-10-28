import glob

import os

import math

import cv2
import numpy as np
import pandas as pd


def write_list_to_file(input_list, file_path, delimiter='\n'):
    with open(file_path, "w") as output:
        output.write(delimiter.join(input_list))


def parse_vatic_annotation_file(annotation_file_path):
    annotations = pd.read_csv(annotation_file_path,
                              sep=' ',
                              header=None,
                              names=['trackid', 'xmin', 'ymin', 'xmax',
                                     'ymax', 'frameid', 'lost', 'occluded',
                                     'generated', 'label'])
    return annotations


def get_prefix0_format_string(item_num):
    """Get the prefix 0 format string from item_num.

    For example, 3 items would result in {:01d}
    """
    max_digit_num = len(str(item_num))
    output_pattern = '{:0' + str(max_digit_num) + 'd}'
    return output_pattern


def rectify_box(xcenter, ycenter, width, height, angle):
    """

    Forward midpoint and backward midpoint are in relation to the angle direction.
    For boxes, the sum of x/y coordinates of opposing points are the same

    :param xcenter:
    :param ycenter:
    :param width:
    :param height:
    :param angle:
    :return:
    """
    radian_angle = math.pi * float(angle) / 180.0 if angle > 0 else math.pi + math.pi * (180.0 - float(angle)) / 180.0
    forward_midpoint = (xcenter + math.cos(radian_angle) * width, ycenter + math.sin(radian_angle) * width)
    backward_midpoint = (xcenter - math.cos(radian_angle) * width, ycenter - math.sin(radian_angle) * width)
    upward_midpoint = (xcenter - math.sin(radian_angle) * height, ycenter + math.cos(radian_angle) * height)
    downward_midpoint = (xcenter + math.sin(radian_angle) * height, ycenter - math.cos(radian_angle) * height)
    upper_right_corner = (
        upward_midpoint[0] + forward_midpoint[0] - xcenter, upward_midpoint[1] + forward_midpoint[1] - ycenter)
    upper_left_corner = (
        upward_midpoint[0] + backward_midpoint[0] - xcenter, upward_midpoint[1] + backward_midpoint[1] - ycenter)
    lower_left_corner = (
        downward_midpoint[0] + backward_midpoint[0] - xcenter, downward_midpoint[1] + backward_midpoint[1] - ycenter)
    lower_right_corner = (
        downward_midpoint[0] + forward_midpoint[0] - xcenter, downward_midpoint[1] + forward_midpoint[1] - ycenter)
    points = np.array([
        [upper_left_corner],
        [upper_right_corner],
        [lower_left_corner],
        [lower_right_corner],
    ], np.float32)
    xmin, ymin, w, h = cv2.boundingRect(points)
    return xmin, ymin, xmin + w - 1, ymin + h - 1


def parse_munich_annotation_file(file_path):
    """
    text format: id type center.x center.y size.width size.height angle

    :param input_dir:
    :return: panda frames
    """
    print("parsing {}".format(file_path))
    imageid = os.path.splitext(os.path.basename(file_path))[0]
    annotations = pd.read_csv(file_path, sep=' ', header=None,
                              names=['unused', 'label', 'xcenter', 'ycenter', 'width', 'height', 'angle'])
    annotations['imageid'] = imageid
    annotations.drop(['unused'], axis=1)
    annotations['xmin'] = 0
    annotations['ymin'] = 0
    annotations['xmax'] = 0
    annotations['ymax'] = 0
    # calculate bounding rect
    for index, row in annotations.iterrows():
        xmin, ymin, xmax, ymax = rectify_box(row['xcenter'], row['ycenter'], row['width'], row['height'], row['angle'])
        annotations.loc[index, "xmin"] = xmin
        annotations.loc[index, "ymin"] = ymin
        annotations.loc[index, "xmax"] = xmax
        annotations.loc[index, "ymax"] = ymax
    return annotations

# parse_annotation_file('nexus/video1/annotations.txt')
