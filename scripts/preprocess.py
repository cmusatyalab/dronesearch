#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cPickle as pickle
import collections
import cv2
import glob
import os
import random
import math
import numpy as np
import shutil

import annotation
import annotation_stats
import fire
import io_util


############################################################
# Preprocess for munich
#
############################################################
def slice_image(im, w, h, slice_is_ratio):
    """Slice image into tiles of width w and height h.

    The returned slice order is horizontal-first.

    Args:
      w: resolution of slice width when slice_is_ratio is False
      h: resolution of slice height when slice_is_ratio is False
      slice_is_ratio: when True, w and h must be >0 and <=1.
      slice_is_ratio: when True, w and h must be >0 and <=1.
    They are the ratio of slice w or h divided by the orginal w or h.
      im: 

    Returns:

    """
    im_h, im_w, _ = im.shape

    if slice_is_ratio:
        # translate ratio into width and height resolution first
        assert (w > 0) and (w <= 1)
        assert (h > 0) and (h <= 1)
        w = int(math.ceil(im_w * w))
        h = int(math.ceil(im_h * h))

    horizontal_num = int(math.ceil(im_w / float(w)))
    vertical_num = int(math.ceil(im_h / float(h)))

    # horizontal idx first, top left is (0,0)
    slices = []
    for h_idx in range(0, horizontal_num):
        vertical_slice = []
        for v_idx in range(0, vertical_num):
            # print('slicing ({}, {})'.format(h_idx, v_idx))
            slice_x = h_idx * w
            slice_y = v_idx * h
            slice_w = min(w, im_w - slice_x)
            slice_h = min(h, im_h - slice_y)
            current_slice = im[slice_y:slice_y + slice_h, slice_x:
                               slice_x + slice_w]
            vertical_slice.append(current_slice)
        slices.append(vertical_slice)
    return slices


def slice_images(input_dir,
                 output_dir,
                 slice_w=225,
                 slice_h=225,
                 slice_is_ratio=False):
    """"Used by Munich dataset to slice images into smaller tiles.

    Args:
      slice_w: resolution of slice width when slice_is_ratio is False (Default
      value = 225)
      slice_h: resolution of slice height when slice_is_ratio is False (Default
      value = 225) slice_is_ratio: when True, slice_w and slice_h must be >0
      and <=1.
    They are treated as the percentage of the original resolution. (Default
    value = False)
      input_dir: Input image directory.
      output_dir: Output image directory.

    Returns:

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_paths = glob.glob(os.path.join(input_dir, '*'))
    for file_path in file_paths:
        # print("slicing file {}".format(file_path))
        output_prefix, ext = os.path.splitext(os.path.basename(file_path))
        im = cv2.imread(file_path)
        slices = slice_image(
            im, slice_w, slice_h, slice_is_ratio=slice_is_ratio)
        # print('total {}x{} slices'.format(len(slices), len(slices[0])))
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


def get_grid_shape(slice_file_paths):
    """Deprecated. Use annotation.py instead

    Args:
      slice_file_paths: 

    Returns:
      and vertically.

    """
    slice_file_basenames = [
        os.path.splitext(os.path.basename(file_path))[0]
        for file_path in slice_file_paths
    ]
    vertical_indices = [
        int(basename.split("_")[-1]) for basename in slice_file_basenames
    ]
    horizontal_indices = [
        int(basename.split("_")[-2]) for basename in slice_file_basenames
    ]
    return max(vertical_indices) + 1, max(horizontal_indices) + 1


def get_slice_shape(slice_file_paths):
    """Deprecated. Use annotation.py instead

    Get the width and the height of a tile

    Args:
      slice_file_paths: 

    Returns: width, height of a tile
      

    """
    sample_file = sorted(slice_file_paths)[0]
    im = cv2.imread(sample_file)
    return im.shape[0], im.shape[1]


def get_slice_contain_roi_bitmap(image_dir, imageid, slice_annotations,
                                 image_annotations):
    """Given the base image annotations, return a roi bitmap for sliced images.

    The bitmap indicates which sliced images has objects of interests

    Args:
      imageid: id for the base image
      annotations: base image annotations
      grid_w: how many slices are there horizontally
      grid_h: how many slices are there vertically
      slice_w: width of the sliced images
      slice_h: height of the slice images
      image_dir: 
      slice_file_paths: 
      image_annotations: 

    Returns:

    """
    grid_h, grid_w = slice_annotations.get_grid_shape(imageid)
    slice_h, slice_w = slice_annotations.get_slice_shape(imageid)

    slice_contain_roi_bitmap = [[False for _ in range(grid_h)]
                                for _ in range(grid_w)]

    for _, image_annotation in image_annotations.iterrows():
        # which cell does this annotation fall into
        xmin, ymin, xmax, ymax = image_annotation["xmin"], \
            image_annotation["ymin"], \
            image_annotation["xmax"], \
            image_annotation["ymax"]
        # upper left, upper right, lower left, lower right
        key_points = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
        grid_coords = set()
        for (x, y) in key_points:
            grid_x, grid_y = int(x / slice_w), int(y / slice_h)
            # due to rectifying bounding boxes to be rectangles,
            # annotations have points that are beyond boundry of image
            # resolutions
            grid_x = min(grid_w - 1, max(grid_x, 0))
            grid_y = min(grid_h - 1, max(grid_y, 0))
            grid_coords.add((grid_x, grid_y))

        # remove those with too small overlap
        roi = (xmin, ymin, xmax, ymax)
        for (grid_x, grid_y) in grid_coords:
            tile = (grid_x * slice_w, grid_y * slice_h, (grid_x + 1) * slice_w,
                    (grid_y + 1) * slice_h)
            if annotation.is_small_bx_in_big_bx(roi, tile):
                print("marking grid cell ({}, {}) as positive".format(
                    grid_x, grid_y))
                slice_contain_roi_bitmap[grid_x][grid_y] = True
    return slice_contain_roi_bitmap


def symlink_or_copy_slices_by_bitmap(slice_file_paths,
                                     slice_contain_roi_bitmap,
                                     category_dir_dict,
                                     copy=False):
    """Copy slices cropped from a base images into dirs of different categories.

    Args:
      sliced_images_path_pattern: string pattern prefix for locating
    all slices from the base image
      slice_contain_roi_bitmap: roi bitmap indicating which slices
    contain what categories
      category_dir_dict: output dir path for different categories
      copy: if True, then copy, otherwise symlink (Default value = False)
      slice_file_paths: 

    Returns:

    """
    grid_h, grid_w = get_grid_shape(slice_file_paths)
    h_idx_format_string = io_util.get_prefix0_format_string(grid_w)
    v_idx_format_string = io_util.get_prefix0_format_string(grid_h)
    idx_format_string = h_idx_format_string + \
        "_" + v_idx_format_string
    slice_file_path_ext = os.path.splitext(slice_file_paths[0])[1]
    slice_file_path_prefix = '_'.join(
        slice_file_paths[0].split('_')[:-2]) + '_'

    for h_idx in range(len(slice_contain_roi_bitmap)):
        for v_idx in range(len(slice_contain_roi_bitmap[h_idx])):
            idx_string = idx_format_string.format(h_idx, v_idx)
            image_file_path = (
                slice_file_path_prefix + idx_string + slice_file_path_ext)
            image_output_dir = category_dir_dict[slice_contain_roi_bitmap[
                h_idx][v_idx]]
            if copy:
                symlink_or_copy_func = shutil.copyfile
            else:
                symlink_or_copy_func = os.symlink
            symlink_or_copy_func(image_file_path,
                                 os.path.join(
                                     image_output_dir,
                                     os.path.basename(image_file_path)))


def group_sliced_images_by_label(dataset, image_dir, annotation_dir,
                                 output_dir):
    """Group images into postivie and negative.

    For current usage, if the cropped image has a car, then it's positive,
    otherwise negative.

    Args:
      image_dir: image_dir with all sliced images
      dataset: 
      annotation_dir: 
      output_dir: 

    Returns:

    """
    slice_annotations = (
        annotation.SliceAnnotationsFactory.get_annotations_for_slices(
            dataset, image_dir, annotation_dir))
    imageids = slice_annotations.imageids
    annotations = slice_annotations.annotations

    group_dir_paths = {
        True: os.path.join(output_dir, "positive"),
        False: os.path.join(output_dir, "negative")
    }
    for group_dir_path in group_dir_paths.values():
        if not os.path.exists(group_dir_path):
            os.makedirs(group_dir_path)

    for imageid in imageids:
        image_annotations = annotations[annotations['imageid'] == imageid]
        slice_file_paths = slice_annotations.get_image_slice_paths(imageid)
        print("processing slices from base image id: {}".format(imageid))
        slice_contain_roi_bitmap = get_slice_contain_roi_bitmap(
            image_dir, imageid, slice_annotations, image_annotations)
        symlink_or_copy_slices_by_bitmap(
            slice_file_paths, slice_contain_roi_bitmap, group_dir_paths)


############################################################
# Preprocess for ImageNet VID
#
############################################################
def gather_images(item_list_file_path,
                  base_dir,
                  output_dir,
                  combine_path_func=os.path.join,
                  max_num=2**20):
    """Gather images from base_dir according to item_list_file_path.

    The gathered images are symlinks in output_dir. If max_num <
    len(item_list), result images are randomly sampled.

    Args:
      item_list_file_path: A file with a relative path at each line.
      base_dir: Base directory to be combined with relative paths.
      combine_path_func: Function to combine base_dir and relative path. The
      input is base_dir, relative_path. The return value could be either a
      single path or a nested list of paths
      output_dir: Output directory
      max_num:  (Default value = 2**20)

    Returns: None

    """
    # get relative paths
    with open(item_list_file_path, 'r') as f:
        contents = f.read().splitlines()
    image_relative_paths = [line.split(' ')[0] for line in contents]

    # get absolute paths
    image_absolute_paths = [
        combine_path_func(base_dir, relative_path)
        for relative_path in image_relative_paths
    ]
    # flatten the list as combine_path_func may return lists as well
    image_absolute_paths = set(io_util.flatten_iterable(image_absolute_paths))
    random.shuffle(image_relative_paths)

    io_util.create_dir_if_not_exist(output_dir)

    selected_num = 0
    for file_path in image_absolute_paths:
        # give each symlink a unique name
        symlink_name = os.path.relpath(file_path, base_dir).replace('/', '_')
        os.symlink(file_path, os.path.join(output_dir, symlink_name))

        selected_num += 1
        if selected_num > max_num:
            break


def gather_frame_sequences(video_list_file_path,
                           vid_base_dir,
                           output_dir,
                           max_num=2**20):
    """Prepare train and test image dir structure.

    Create symlinks in output_dir pointing to image paths specified by
    video_list_file_path.

    Args:
      video_list_file_path: text file with relative image path at each line
      vid_base_dir: image base dir. combined with each line in
    video_list_file_path to get absolute path.
      max_num: max number of selection (Default value = 2**20)
      output_dir: Output directory

    Returns: None

    """

    def combine_path_func(base_dir, relative_path):
        frame_sequence_dir_path = os.path.join(base_dir, relative_path)
        file_paths = glob.glob(os.path.join(frame_sequence_dir_path, '*'))
        return file_paths

    gather_images(
        video_list_file_path,
        vid_base_dir,
        output_dir,
        combine_path_func=combine_path_func,
        max_num=2**20)


############################################################
# Preprocess for stanford
#
############################################################
def sample_files_from_directory(input_dir, output_dir, sample_num=1000):
    """Sample by copy files from input_dir to output_dir.

    Args:
      input_dir: 
      output_dir: 
      sample_num:  (Default value = 1000)

    Returns:

    """
    file_paths = glob.glob(os.path.join(input_dir, '*'))
    random.shuffle(file_paths)
    sample_paths = file_paths[:sample_num]
    io_util.create_dir_if_not_exist(output_dir)
    for sample_path in sample_paths:
        os.symlink(sample_path,
                   os.path.join(output_dir, os.path.basename(sample_path)))


def flatten_stanford_dir(input_dir, output_dir, followlinks=False):
    """Move all videos in the stanford dataset to the same dir.

    Args:
      input_dir: 
      output_dir: 
      followlinks:  (Default value = False)

    Returns:

    """
    io_util.flatten_directory_with_symlink(input_dir, output_dir, followlinks)


def group_stanford_images(image_dir,
                          annotation_dir,
                          output_dir,
                          max_positive_num=10000,
                          max_negative_num=20000):
    """Group stanford images into postivie and negative.
    
    If the image has a car/bus, then it's positive, otherwise negative.

    Args:
      image_dir: should be the either the "train" or "test" dir with
    first level being video names and second level images.
      annotation_dir: 
      output_dir: 
      max_positive_num:  (Default value = 10000)
      max_negative_num:  (Default value = 20000)

    Returns:

    """
    VIDEO_DIR_SUFFIX = '_video.mov'
    IMAGE_EXT = '.jpg'

    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    print('loaded annotations.')
    video_list = os.listdir(image_dir)
    videoids = [
        videoname.replace('_video.mov', '') for videoname in video_list
    ]
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
        mask = annotation.get_positive_annotation_mask(annotations)
        target_annotations = annotations[mask].copy()
        # groupby videoid and frameid to remove duplicates
        unique_target_annotations = target_annotations.groupby(
            ['videoid', 'frameid']).sum()
        # list of (video ids, frame ids)
        unique_target_tuples = unique_target_annotations.index.values
        if category == 'negative':
            # when negative, we override the unique_target_tuples with the
            # negation of target tuples
            unique_annotations = annotations.groupby(['videoid',
                                                      'frameid']).sum()
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
            relpath = os.path.relpath(src_image_path, image_dir)
            dst_relpath = relpath.replace(os.path.sep, '_')
            dst_image_path = os.path.join(category_output_dir, dst_relpath)
            os.symlink(src_image_path, dst_image_path)


def split_train_test(data_list_file, output_dir, test_percentage=0.1):
    """Split train test set.

    Args:
      data_list_file: a file with each data occupying a line.
      output_dir: output_dir that contains train.txt and test.txt
      test_percentage:  (Default value = 0.1)

    Returns:

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
    io_util.write_list_to_file(train_list, os.path.join(
        output_dir, 'train.txt'))
    io_util.write_list_to_file(test_list, os.path.join(output_dir, 'test.txt'))


def rename_okutama_video_file_to_match_id(video_dir):
    """Rename okutama video files to match their ids in the annotation.
    e.g.
    Drone1_Morning_1.1.10.mp4 --> 1.1.10.
    """
    video_file_names = os.listdir(video_dir)
    for video_file_name in video_file_names:
        if '_' not in video_file_name:
            raise ValueError(
                'Wrong file name format: {}'.format(video_file_name))
        file_name_without_ext = os.path.splitext(video_file_name)[0]
        output_file_name = file_name_without_ext.split('_')[2]
        os.rename(
            os.path.join(video_dir, video_file_name),
            os.path.join(video_dir, output_file_name))


def rename_elephant_video_file_to_match_id(video_dir):
    """Rename elephant video files to match their ids.
    e.g.
    01.mp4 --> 01
    """
    video_file_names = os.listdir(video_dir)
    for video_file_name in video_file_names:
        file_name_without_ext = os.path.splitext(video_file_name)[0]
        output_file_name = file_name_without_ext
        os.rename(
            os.path.join(video_dir, video_file_name),
            os.path.join(video_dir, output_file_name))


def rename_stanford_image_dir_to_match_id(image_dir):
    """Rename stanford image dir to match their ids in the annotation.
    e.g.
    gates_video0_video.mov --> gates_video0
    """
    video_file_names = os.listdir(image_dir)
    for video_file_name in video_file_names:
        if '_' not in video_file_name:
            raise ValueError(
                'Wrong file name format: {}'.format(video_file_name))
        file_name_without_ext = os.path.splitext(video_file_name)[0]
        output_file_name = '_'.join(file_name_without_ext.split('_')[:2])
        os.rename(
            os.path.join(image_dir, video_file_name),
            os.path.join(image_dir, output_file_name))


def resize_okutama_frame_sequence(video_dir,
                                  output_width=3808,
                                  output_height=2240):
    """Resize okutama frame sequence in place so that they are multiples of 224.

    """
    video_file_paths = glob.glob(os.path.join(video_dir, '*', '*'))
    for video_file_path in video_file_paths:
        im = cv2.imread(video_file_path)
        resized_im = cv2.resize(im, (output_width, output_height))
        cv2.imwrite(video_file_path, resized_im)


def resize_frame_sequence_by_id(video_dir, output_dir, video_id, long_edge,
                                short_edge):
    """Resize frame sequence in place.

    """
    if type(video_id) is int:
        video_id = '{:02d}'.format(video_id)
    frame_sequence_name = video_id
    print('working on {}'.format(frame_sequence_name))
    output_frame_sequence_dir = os.path.join(output_dir, frame_sequence_name)
    io_util.create_dir_if_not_exist(output_frame_sequence_dir)
    video_file_paths = glob.glob(
        os.path.join(video_dir, frame_sequence_name, '*'))

    for video_file_path in video_file_paths:
        dst_file_path = os.path.join(output_frame_sequence_dir,
                                     os.path.basename(video_file_path))
        im = cv2.imread(video_file_path)
        im_h, im_w, _ = im.shape
        if im_h > im_w:
            resized_im = cv2.resize(im, (short_edge, long_edge))
        else:
            resized_im = cv2.resize(im, (long_edge, short_edge))
        cv2.imwrite(dst_file_path, resized_im)


def resize_stanford_frame_sequence(video_dir,
                                   output_dir,
                                   split_name='test',
                                   long_edge=1792,
                                   short_edge=896):
    """Resize stanford frame sequence in place.

    """
    if split_name == 'test':
        frame_sequence_names = annotation_stats.stanford_test_videos
    else:
        frame_sequence_names = annotation_stats.stanford_train_videos

    for frame_sequence_name in frame_sequence_names:
        output_frame_sequence_dir = os.path.join(output_dir,
                                                 frame_sequence_name)
        io_util.create_dir_if_not_exist(output_frame_sequence_dir)
        video_file_paths = glob.glob(
            os.path.join(video_dir, frame_sequence_name, '*'))

        for video_file_path in video_file_paths:
            dst_file_path = os.path.join(output_frame_sequence_dir,
                                         os.path.basename(video_file_path))
            im = cv2.imread(video_file_path)
            im_h, im_w, _ = im.shape
            if im_h > im_w:
                resized_im = cv2.resize(im, (short_edge, long_edge))
            else:
                resized_im = cv2.resize(im, (long_edge, short_edge))
            cv2.imwrite(dst_file_path, resized_im)


OKUTAMA_TRAIN_VIDEOS = [
    '1.1.3', '1.1.2', '1.1.5', '1.1.4', '2.2.7', '2.1.7', '2.1.4', '2.2.11',
    '1.1.10'
]
OKUTAMA_TEST_VIDEOS = ['2.2.2', '2.2.4', '1.1.7']


def sample_train_test_frames(tile_classification_annotation_dir,
                             sample_num_per_video, output_file_path,
                             video_ids):
    print('loading tile annotations from {}'.format(
        tile_classification_annotation_dir))

    total_sample_ids = collections.defaultdict(list)
    for video_id in video_ids:
        image_id_to_classification_label = (
            io_util.load_all_pickles_from_dir(
                tile_classification_annotation_dir, video_ids=[video_id]))
        positive_image_ids = ([
            k for k, v in image_id_to_classification_label.items() if v
        ])
        negative_image_ids = ([
            k for k, v in image_id_to_classification_label.items() if not v
        ])
        print(
            '{} total tiles: {}, total positive images: {}, total negative images {}'.
            format(video_id,
                   len(positive_image_ids) + len(negative_image_ids),
                   len(positive_image_ids), len(negative_image_ids)))

        random.shuffle(positive_image_ids)
        random.shuffle(negative_image_ids)
        sample_num_cur_video = np.min([
            sample_num_per_video,
            len(positive_image_ids),
            len(negative_image_ids)
        ])
        print('sampling {} images for each class'.format(sample_num_cur_video))
        total_sample_ids['positive'].extend(
            positive_image_ids[:sample_num_cur_video])
        total_sample_ids['negative'].extend(
            negative_image_ids[:sample_num_cur_video])

    output_dir = os.path.dirname(output_file_path)
    io_util.create_dir_if_not_exist(output_dir)
    with open(output_file_path, 'wb') as f:
        pickle.dump(total_sample_ids, f)


def sample_okutama_frames(tile_classification_annotation_dir,
                          sample_num_per_video,
                          output_dir,
                          split_name='train'):
    """Sample okutama frame ids for train and test.

    Only frame ids are sampled based on annotation, not real images.

    Args:
      tile_classification_annotation_dir: 
      sample_num_per_video: # of samples per class. 2x this number for total samples for classification.
      output_dir: 
      split_name:  (Default value = 'train')

    Returns:

    """
    assert split_name in ['train', 'test']

    if split_name == 'train':
        video_ids = annotation_stats.okutama_train_videos
    else:
        video_ids = annotation_stats.okutama_test_videos
    output_file_path = os.path.join(output_dir, '{}.pkl'.format(split_name))
    sample_train_test_frames(
        tile_classification_annotation_dir,
        sample_num_per_video,
        output_file_path,
        video_ids=video_ids)


def sample_stanford_frames(tile_classification_annotation_dir,
                           sample_num_per_video,
                           output_dir,
                           split_name='train'):
    """Sample stanford frame ids for train and test.

    Only frame ids are sampled based on annotation, not real images.

    Args:
      tile_classification_annotation_dir: 
      sample_num_per_video: # of samples per class. 2x this number for total samples for classification.
      output_dir: 
      split_name:  (Default value = 'train')

    Returns:

    """
    assert split_name in ['train', 'test']

    if split_name == 'train':
        video_ids = annotation_stats.stanford_train_videos
    else:
        video_ids = annotation_stats.stanford_test_videos
    output_file_path = os.path.join(output_dir, '{}.pkl'.format(split_name))
    sample_train_test_frames(
        tile_classification_annotation_dir,
        sample_num_per_video,
        output_file_path,
        video_ids=video_ids)


def sample_dataset_frames(dataset_name,
                          tile_classification_annotation_dir,
                          sample_num_per_video,
                          output_dir,
                          split_name='train'):
    """Sample dataset frame ids for train and test.

    Only frame ids are sampled based on annotation, not real images.

    Args:
    dataset name
      tile_classification_annotation_dir: 
      sample_num_per_video: # of samples per class. 2x this number for total samples for classification.
      output_dir: 
      split_name:  (Default value = 'train')

    Returns:

    """
    assert dataset_name in annotation_stats.dataset.keys()
    assert split_name in ['train', 'test']

    video_ids = annotation_stats.dataset[dataset_name][split_name]
    output_file_path = os.path.join(output_dir, '{}.pkl'.format(split_name))
    sample_train_test_frames(
        tile_classification_annotation_dir,
        sample_num_per_video,
        output_file_path,
        video_ids=video_ids)


if __name__ == "__main__":
    fire.Fire()
