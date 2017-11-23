from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import math
import numpy as np
import os
import functools

import pandas as pd


def write_list_to_file(input_list, file_path, delimiter='\n'):
    with open(file_path, "w") as output:
        output.write(delimiter.join(input_list))


def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def flatten_iterable(iterable):
    """Flatten an iterable whose elements may also be iterable.

    e.g. flatten a nested list

    Args:
      iterable: Input iterable to flatten

    Returns: A generator with flattened iterable

    """
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in flatten_iterable(e):
                yield f
        else:
            yield e


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
    import cv2
    radian_angle = math.pi * float(
        angle) / 180.0 if angle > 0 else math.pi + math.pi * (
            180.0 - float(angle)) / 180.0
    forward_midpoint = (xcenter + math.cos(radian_angle) * width,
                        ycenter + math.sin(radian_angle) * width)
    backward_midpoint = (xcenter - math.cos(radian_angle) * width,
                         ycenter - math.sin(radian_angle) * width)
    upward_midpoint = (xcenter - math.sin(radian_angle) * height,
                       ycenter + math.cos(radian_angle) * height)
    downward_midpoint = (xcenter + math.sin(radian_angle) * height,
                         ycenter - math.cos(radian_angle) * height)
    upper_right_corner = (upward_midpoint[0] + forward_midpoint[0] - xcenter,
                          upward_midpoint[1] + forward_midpoint[1] - ycenter)
    upper_left_corner = (upward_midpoint[0] + backward_midpoint[0] - xcenter,
                         upward_midpoint[1] + backward_midpoint[1] - ycenter)
    lower_left_corner = (downward_midpoint[0] + backward_midpoint[0] - xcenter,
                         downward_midpoint[1] + backward_midpoint[1] - ycenter)
    lower_right_corner = (downward_midpoint[0] + forward_midpoint[0] - xcenter,
                          downward_midpoint[1] + forward_midpoint[1] - ycenter)
    points = np.array([
        [upper_left_corner],
        [upper_right_corner],
        [lower_left_corner],
        [lower_right_corner],
    ], np.float32)
    xmin, ymin, w, h = cv2.boundingRect(points)
    return xmin, ymin, xmin + w - 1, ymin + h - 1


def load_annotation_from_dir(annotation_dir, parse_func):
    """Load all annotation files in annotation_dir into a single data frame."""
    cache_file_path = os.path.join(annotation_dir, 'cache.pkl')
    if os.path.exists(cache_file_path):
        print('find cache file at: {}. Using cached annotations'.format(
            cache_file_path))
        all_annotation = pd.read_pickle(cache_file_path)
    else:
        print(('No cache found. Read from annotation files '
               'directly and writing caches...'))
        file_paths = glob.glob(
            os.path.join(os.path.abspath(annotation_dir), "*"))
        annotation_by_file = []
        for file_path in file_paths:
            file_annotation = parse_func(file_path)
            annotation_by_file.append(file_annotation)
        all_annotation = pd.concat(annotation_by_file, ignore_index=True)
        all_annotation.to_pickle(cache_file_path)
    return all_annotation


def load_munich_annotation(annotation_dir):
    """Load all munich annotations into a single data frame."""
    return load_annotation_from_dir(annotation_dir,
                                    parse_munich_annotation_file)


def load_stanford_campus_annotation(annotation_dir):
    """Load all stanford campus annotations into a single data frame."""
    return load_annotation_from_dir(annotation_dir,
                                    parse_vatic_annotation_file)


def load_okutama_annotation(annotation_dir):
    """Load all stanford campus annotations into a single data frame."""
    column_names = [
        'trackid', 'xmin', 'ymin', 'xmax', 'ymax', 'frameid', 'lost',
        'occluded', 'generated', 'label', 'action'
    ]
    return load_annotation_from_dir(annotation_dir,
                                    functools.partial(
                                        parse_vatic_annotation_file,
                                        column_names=column_names))


def load_stanford_video_ids_from_file(video_list_file_path):
    """Load video ids from video list file. The file has video path (with
_video.move) at each line of its content

    Args:
      video_list_file_path: 

    Returns:

    """
    with open(video_list_file_path, 'r') as f:
        video_names = f.read().splitlines()
    videoids = [
        video_name.replace('_video.mov', '') for video_name in video_names
    ]
    return videoids


def stanford_video_id_to_frame_sequence_dir(video_id):
    return '{}_video.mov'.format(video_id)


def parse_vatic_annotation_file(file_path,
                                column_names=[
                                    'trackid', 'xmin', 'ymin', 'xmax', 'ymax',
                                    'frameid', 'lost', 'occluded', 'generated',
                                    'label'
                                ]):
    print("parsing {}".format(file_path))
    annotations = pd.read_csv(
        file_path, sep=' ', header=None, names=column_names)
    videoid = os.path.splitext(os.path.basename(file_path))[0]
    videoid = videoid.replace('_annotations', '')
    annotations['videoid'] = videoid
    return annotations


def parse_munich_annotation_file(file_path):
    """
    text format: id type center.x center.y size.width size.height angle

    :param input_dir:
    :return: panda frames
    """
    print("parsing {}".format(file_path))
    imageid = os.path.splitext(os.path.basename(file_path))[0]
    annotations = pd.read_csv(
        file_path,
        sep=' ',
        header=None,
        names=[
            'unused', 'label', 'xcenter', 'ycenter', 'width', 'height', 'angle'
        ])
    annotations['imageid'] = imageid
    annotations.drop(['unused'], axis=1)

    annotations['xmin'] = 0
    annotations['ymin'] = 0
    annotations['xmax'] = 0
    annotations['ymax'] = 0
    # the original ground truth is given with boxes that are not parallel to image.
    # need to transform them into bounding rects to follow object detection ground truth convention
    # note these annotation may go beyond image resolution due to rectification
    for index, row in annotations.iterrows():
        xmin, ymin, xmax, ymax = rectify_box(row['xcenter'], row['ycenter'],
                                             row['width'], row['height'],
                                             row['angle'])
        annotations.loc[index, 'xmin'] = xmin
        annotations.loc[index, 'ymin'] = ymin
        annotations.loc[index, 'xmax'] = xmax
        annotations.loc[index, 'ymax'] = ymax
    return annotations


def flatten_directory_with_symlink(input_dir, output_dir, followlinks):
    """Flatten the input_dir into only 1 level.

    File names uses sub directory name as prefix for naming.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, filenames in os.walk(input_dir, followlinks=followlinks):
        for filename in filenames:
            src_filepath = os.path.join(root, filename)
            relpath = os.path.relpath(src_filepath, input_dir)
            dst_relpath = relpath.replace(os.path.sep, '_')
            dst_filepath = os.path.join(output_dir, dst_relpath)
            print("symlink {} -> {}".format(dst_filepath, src_filepath))
            os.symlink(src_filepath, dst_filepath)


# parse_annotation_file('nexus/video1/annotations.txt')
