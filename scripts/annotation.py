"""Annotation Wrapper and related logic.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import abc
import cPickle as pickle
import collections
import cv2
import functools
import glob
import itertools
import json
import numpy as np
import operator
import os

import fire
import io_util
import redis
from annotation_stats import (
    okutama_test_videos, okutama_train_videos,
    okutama_video_id_to_original_resolution, okutama_video_to_frame_num,
    stanford_test_videos, stanford_train_videos, stanford_video_to_frame_num,
    stanford_video_id_to_original_resolution)


def get_positive_annotation_mask(annotations, labels=['Car', 'Bus']):
    """Car/NoCar annotation pandas dataframe mask."""
    lost_mask = (annotations['lost'] == 0)
    label_mask = annotations['label'].isin(labels)
    mask = label_mask & lost_mask
    return mask


def _intersection_area(bx1, bx2):
    """Calculate intersection areas among two bounding boxes.

    Args:
      bx1: Bounding box1 in the form of (xmin, ymin, xmax, ymax)
      bx2: Bounding box2 in the same format

    Returns:

    """
    ixmin = np.maximum(bx1[0], bx2[0])
    iymin = np.maximum(bx1[1], bx2[1])
    ixmax = np.minimum(bx1[2], bx2[2])
    iymax = np.minimum(bx1[3], bx2[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    return inters


def is_small_bx_in_big_bx(
        small_bx, large_bx,
        intersection_over_small_bx_percentage_threshold=0.25):
    """Check if the small bounding box is in the big bounding box.

    The condition is determined by the intersection area over the area of the
    small bounding box. This method is used to determine whether ROI is in a
    tile or not.

    Args:
      small_bx: Small bounding box in the form of (xmin, ymin, xmax, ymax)
      large_bx: Large bounding box in the same form.
      intersection_over_small_bx_percentage_threshold: An percentage above this
      threshold would be considered the small box to be in the big box (Default
      value = 0.25)

    Returns: Whether or not the small box is in the big box

    """
    intersection_area = _intersection_area(small_bx, large_bx)
    small_bx_area = (small_bx[2] - small_bx[0] + 1.0) * (
        small_bx[3] - small_bx[1] + 1.0)
    intersection_over_small_bx_percentage = (
        intersection_area * 1.0 / small_bx_area)
    ret = (intersection_over_small_bx_percentage >
           intersection_over_small_bx_percentage_threshold)
    return ret


class SliceAnnotations(object):
    """Annotations base class for tiled experiments.

    Used by group_sliced_images_by_label to get annotations of interests.
    """

    def __init__(self, dataset, slice_image_dir, annotation_dir):
        self._dataset = dataset
        self._annotation_dir = annotation_dir
        self._slice_image_dir = slice_image_dir
        # annotations for full resolution images
        self._base_annotations = None

    @abc.abstractproperty
    def imageids(self):
        """Return imageids of interets"""

    @abc.abstractproperty
    def annotations(self):
        """Return annotations of interests"""

    @abc.abstractmethod
    def get_sliced_images_path_pattern(self, imageid):
        """Return sliced image path patterns for the imageid"""

    @abc.abstractmethod
    def get_image_slice_paths(self, imageid):
        """Get absolute file paths of all tile images from the base image."""

    @abc.abstractmethod
    def get_grid_shape(self, imageid):
        """Get how many tiles a base image is divided into."""

    @abc.abstractmethod
    def get_slice_shape(self, imageid):
        """Get the shape of a tile image."""


class SliceAnnotationsFactory(object):
    """Factory for tile annotations
    """

    @classmethod
    def get_annotations_for_slices(*args, **kwargs):
        # first argument is the class
        args = args[1:]
        dataset = args[0]
        if dataset == 'munich':
            return MunichCarSliceAnnotations(*args, **kwargs)
        elif dataset == 'stanford':
            return StanfordCarSliceAnnotations(*args, **kwargs)
        else:
            raise NotImplementedError(
                "Dataset {} is not supported".format(dataset))


class MunichCarSliceAnnotations(SliceAnnotations):
    """Annotations for car/nocar classification on Munich dataset.

    The annotation_dir for munich dataset is already different for train and
    test data. Therefore, the _base_annotations are the annotations we're
    interested in, since no sampling on the train/test dataset is used.
    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # annotations for full resolution images
        self._base_annotations = io_util.load_munich_annotation(
            self._annotation_dir)

    @property
    def imageids(self):
        return set(self._annotations['imageid'].tolist())

    @property
    def annotations(self):
        return self._base_annotations

    def get_sliced_images_path_pattern(self, imageid):
        return os.path.join(self._slice_image_dir, '*{}*'.format(imageid))


class StanfordCarSliceAnnotations(SliceAnnotations):
    """Annotations for car/nocar classification on Stanford dataset.

    The annotation_dir for stanford dataset loads all annotations. Filtering
    are performed to return only the imageids and annotations of interests.

    Args:

    Returns: A tiled annotation objects for Stanford dataset

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # annotations for full resolution images
        self._base_annotations = io_util.load_stanford_campus_annotation(
            self._annotation_dir)
        self._imageids = None
        self._annotations = None
        self._imageids_to_tile_paths_lut = collections.defaultdict(list)
        print('building imageids to tile paths map')
        self._load_imageids_and_imageids_to_tile_paths_lut()
        print('finished loading annotations')

    def _load_imageids_and_imageids_to_tile_paths_lut(self):
        """Populate imageids and imageids to tile path map"""
        slice_file_paths = glob.glob(os.path.join(self._slice_image_dir, '*'))
        for file_path in slice_file_paths:
            slice_file_basename = os.path.basename(file_path)
            components = slice_file_basename.split('_')
            imageid = '_'.join(
                [components[0], components[1],
                 str(int(components[3]))])
            self._imageids_to_tile_paths_lut[imageid].append(file_path)
        self._imageids = self._imageids_to_tile_paths_lut.keys()

    @property
    def imageids(self):
        """Return the entire set of image ids for the loaded annotations"""
        if not self._imageids:
            self._load_imageids_and_imageids_to_paths_lut()
        return self._imageids

    @property
    def annotations(self):
        """Return the annotations loaded"""
        if not self._annotations:
            self._base_annotations['imageid'] = (
                self._base_annotations['videoid'] + '_' +
                self._base_annotations['frameid'].astype(str))
            car_annotation_mask = get_positive_annotation_mask(
                self._base_annotations)
            video_id_mask = self._base_annotations['imageid'].isin(
                self.imageids)
            mask = car_annotation_mask & video_id_mask
            self._annotations = self._base_annotations[mask]
        return self._annotations

    def get_imageid(self, videoid, frameid):
        """Construct annotation imageid from dataframe videoid and frameid.

        Args:
          videoid: A value from videoid column from annotation dataframe
          frameid: A value from frameid column from annotation dataframe

        Returns:

        """
        return '{}_{}'.format(videoid, frameid)

    def get_tile_file_basename_without_ext(self, imageid, tile_coord):
        """Construct the tile image base filename without extension.

        The base filename without extension serves as the redis key for
        indexing prediction results.

        Args:
          imageid: Annotation imageid
          tile_coord: Coordinate of a tile in the form of (x, y)

        Returns:

        """
        videoid, frameid = self._convert_imageid_to_videoid_and_frameid(
            imageid)
        return '{}_video.mov_{:010d}_{}_{}'.format(
            videoid, frameid, tile_coord[0], tile_coord[1])

    def _convert_imageid_to_videoid_and_frameid(self, imageid):
        """Convert an annotation imageid to videoid and frameid

        Videoid corresponds to the 'videoid' column in annotation pandas
        dataframe. And frameid corresponds to 'frameid'

        Args:
          imageid: Imageid of the base image.

        Returns: videoid, frameid

        """
        components = imageid.split('_')
        videoid = '_'.join([components[0], components[1]])
        frameid = int(components[2])
        return videoid, frameid

    def get_sliced_images_path_pattern(self, imageid):
        """Get the search string pattern for tile images divided from a base image.

        Args:
          imageid: Imageid of the base image.

        Returns: The search string.

        """
        videoid, frameid = self._convert_imageid_to_videoid_and_frameid(
            imageid)
        return os.path.join(self._slice_image_dir,
                            '*{}_video.mov_{:010d}_*'.format(videoid, frameid))

    def get_image_slice_paths(self, imageid):
        """Get absolute file paths of all tile images from the base image.

        Args:
          imageid: Imageid of the base image.

        Returns: A list of absolute tile image file path.

        """
        # sliced_images_path_pattern = self.get_sliced_images_path_pattern(
        #     imageid)
        # slice_file_paths = glob.glob(sliced_images_path_pattern)
        slice_file_paths = self._imageids_to_tile_paths_lut[imageid]
        if not slice_file_paths:
            raise ValueError(
                'Did not find any slices for imageid: {}'.format(imageid))
        return slice_file_paths

    def get_grid_shape(self, imageid):
        """Get how many tiles a base image is divided into.

        Args:
          imageid: Imageid of the base image.

        Returns: Height and width of the divided grid.

        """
        slice_file_paths = self.get_image_slice_paths(imageid)
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

    def get_slice_shape(self, imageid):
        """Get the shape of a tile image.

        Args:
          imageid: Imageid of the base image.

        Returns: Height and width of the tile.

        """
        slice_file_paths = self.get_image_slice_paths(imageid)
        sample_file = slice_file_paths[0]
        im = cv2.imread(sample_file)
        return im.shape[0], im.shape[1]

    def get_tiles_contain_bounding_box(self, imageid, bx):
        """Return coordinates of the tiles that contain input bounding box.

        The tile coordinate system is the same as opencv image coordinate
        system, which starts with (0,0) for the top-left corner. First element
        of coordinate is x-axis, with direction going to the right. The second
        element of the coordinate is y-axis, with the direction going down. For
        an images divided into 2x2 tiles, the coordinates are following:
        (0,0) | (1,0)
        -------------
        (0,1) | (1,1)

        Args:
          imageid: Base image id. should be a value 'imageid' column when
          loaded annotations
          bx: Bounding box tuple in the form of (xmin, ymin, xmax, ymax).

        Returns: A list of tile coordinates that overlap with this bounding
        box.

        Args:
          imageid: 
          bx: 

        Returns:


        """
        grid_h, grid_w = self.get_grid_shape(imageid)
        slice_h, slice_w = self.get_slice_shape(imageid)

        (xmin, ymin, xmax, ymax) = bx
        # upper left, upper right, lower left, lower right
        key_points = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]

        tiles = set()
        for (x, y) in key_points:
            grid_x, grid_y = int(x / slice_w), int(y / slice_h)

            # due to rectifying bounding boxes to be rectangles,
            # annotations have points that are beyond boundry of image
            # resolutions
            grid_x = min(grid_w - 1, max(grid_x, 0))
            grid_y = min(grid_h - 1, max(grid_y, 0))
            tiles.add((grid_x, grid_y))
        return list(tiles)


def get_continuous_sequence(frameids):
    """Get list of consecutive sequence.

    Args:
      frameids: A list of frame ids, e.g. [1,2,3,5,6,7]

    Returns: A list of consecutive frame id list, e.g.[[1,2,3],[5,6,7]]

    """
    event_lengths = []
    # there are a few tracks in which there are 'lost' frames in between
    for k, g in itertools.groupby(enumerate(frameids), lambda (i, x): i - x):
        event_frameids = map(operator.itemgetter(1), g)
        event_lengths.append(len(event_frameids))
    return event_lengths


def group_annotation_by_unique_track_ids(annotations):
    """Group annotations by unique track ids. track_id in stanford dataset is only
    unique within a video. This method create unique_track_id by combine it
    with video_id.

    Args:
      annotations: 

    Returns:

    """
    annotations['unique_track_id'] = (
        annotations['videoid'] + '_' + annotations['trackid'].astype(str))
    track_annotations_grp = annotations.groupby(['unique_track_id'])
    return track_annotations_grp


def filter_annotation_by_label(annotations, labels=['Car', 'Bus']):
    mask = get_positive_annotation_mask(annotations, labels=labels)
    target_annotations = annotations[mask].copy()
    return target_annotations


def filter_annotation_by_video_name(annotations, video_list_file_path):
    print('filter annotation by videos specified by {}'.format(
        video_list_file_path))
    videoids = io_util.load_stanford_video_ids_from_file(video_list_file_path)
    assert len(videoids) > 0
    # filter by video name
    annotations = annotations[annotations['videoid'].isin(videoids)]
    return annotations


def print_video_id_with_car_events(annotation_dir):
    """Print car events stats in the stanford dataset.

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """
    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    annotations = filter_annotation_by_label(annotations)
    video_ids = list(set(annotations['videoid']))
    video_ids.sort()
    print('There are {} videos with car events.'.format(len(video_ids)))
    print(json.dumps(video_ids, indent=4))


def print_car_event_stats(annotation_dir, video_list_file_path=None):
    """Print car events stats in the stanford dataset.

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """
    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    if video_list_file_path:
        annotations = filter_annotation_by_video_name(annotations,
                                                      video_list_file_path)
    annotations = filter_annotation_by_label(annotations)
    track_annotations_grp = group_annotation_by_unique_track_ids(annotations)
    # key: video, value: dict of {track_id: list of event lengths}
    video_to_events = collections.defaultdict(dict)
    video_to_events_length = collections.defaultdict(dict)
    for track_id, track_annotations in track_annotations_grp:
        video_id = '_'.join(track_id.split('_')[:2])
        sorted_track_annotations = track_annotations.sort_values('frameid')
        frameids = sorted_track_annotations['frameid'].values
        if len(frameids) < 20:
            # some annotations are duplicates that should have been removed.
            # usually a label is a car another label is a bus, which overlaps
            # for <1s
            continue
        video_to_events[video_id][track_id] = set(frameids)
        video_to_events_length[video_id][track_id] = len(frameids)
    print('events in videos: \n {}'.format(
        json.dumps(video_to_events_length, indent=4)))
    total_events = sum(
        [len(track.values()) for track in video_to_events_length.values()])
    print('total events: {}'.format(total_events))

    print('video positive frame breakdown: {}'.format(total_events))
    video_to_positive_frame_num = {}
    for video_id, track in video_to_events.items():
        video_positive_frame_ids = set()
        for track_id, frame_ids in track.items():
            video_positive_frame_ids |= frame_ids
        video_to_positive_frame_num[video_id] = len(video_positive_frame_ids)
    print(json.dumps(video_to_positive_frame_num, indent=4))
    print('total positive frames {}'.format(
        sum(video_to_positive_frame_num.values())))


def store_positive_frame_id_by_video_to_redis(annotation_dir,
                                              redis_db,
                                              video_list_file_path=None):
    """Store positive frameids to redis. Key is video id, value is frame id

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """
    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    if video_list_file_path:
        annotations = filter_annotation_by_video_name(annotations,
                                                      video_list_file_path)
    r_server = redis.StrictRedis(host='localhost', port=6379, db=redis_db)
    annotations = filter_annotation_by_label(annotations)
    track_annotations_grp = group_annotation_by_unique_track_ids(annotations)
    video_to_frame_ids = collections.defaultdict(set)
    for track_id, track_annotations in track_annotations_grp:
        print('analyzing track: {}'.format(track_id))
        video_id = '_'.join(track_id.split('_')[:2])
        sorted_track_annotations = track_annotations.sort_values('frameid')
        frame_ids = sorted_track_annotations['frameid'].values
        if len(frame_ids) < 20:
            # some annotations are duplicates that should have been removed.
            # usually a label is a car another label is a bus, which overlaps
            # for <1s
            continue
        video_to_frame_ids[video_id] |= set(frame_ids)
    for video_id, frame_ids in video_to_frame_ids.items():
        print('{} has {} positive frames'.format(video_id, len(frame_ids)))
        id_list = map(int, list(frame_ids))
        id_list.sort()
        r_server.rpush(video_id, *id_list)


def store_annotation_by_frame_id_to_redis(annotation_dir,
                                          redis_db,
                                          redis_port=6379,
                                          video_list_file_path=None):
    """Store annotation by frame id to redis.

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """
    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    if video_list_file_path:
        annotations = filter_annotation_by_video_name(annotations,
                                                      video_list_file_path)
    r_server = redis.StrictRedis(
        host='localhost', port=redis_port, db=redis_db)
    annotations = filter_annotation_by_label(annotations)
    total_rows = annotations.shape[0]
    current_rows = 0
    for _, image_annotation in annotations.iterrows():
        xmin, ymin, xmax, ymax = image_annotation['xmin'], \
            image_annotation['ymin'], \
            image_annotation['xmax'], \
            image_annotation['ymax']
        frame_id = 'gt_' + image_annotation['videoid'] + '_' + str(
            image_annotation['frameid'])
        r_server.rpush(frame_id, json.dumps((xmin, ymin, xmax, ymax)))
        current_rows += 1
        if current_rows % 100 == 0:
            print('finished [{}/{}]'.format(current_rows, total_rows))
    print('finished [{}/{}]'.format(current_rows, total_rows))


def print_stanford_car_events(annotation_dir):
    """Print car events stats in the stanford dataset.

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """

    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    annotations = filter_annotation_by_label(
        annotations, labels=['Car', 'Bus'])
    video_to_event_frames = collections.defaultdict(dict)
    video_ids = list(set(annotations['videoid']))
    for video_id in video_ids:
        video_annotations = annotations[annotations['videoid'] == video_id]
        video_positive_frame_ids = list(set(video_annotations['frameid']))
        total_frame_num = stanford_video_to_frame_num[video_id]
        video_to_event_frames[video_id] = {
            'positive_frame_num':
            len(video_positive_frame_ids),
            'negative_frame_num':
            max(0, total_frame_num - len(video_positive_frame_ids)),
            'total_frame_num':
            total_frame_num,
            'positive_frame_percent':
            len(video_positive_frame_ids) * 1.0 / total_frame_num,
        }
    print('There are {} videos with car events.'.format(len(video_ids)))
    print(json.dumps(video_to_event_frames, indent=4))

    video_positive_nums = [
        video_to_event_frames[video_id]['positive_frame_num']
        for video_id in video_to_event_frames.keys()
    ]
    video_negative_nums = [
        video_to_event_frames[video_id]['negative_frame_num']
        for video_id in video_to_event_frames.keys()
    ]
    print('total positive frame num: {}'.format(np.sum(video_positive_nums)))
    print('total negative frame num: {}'.format(np.sum(video_negative_nums)))

    print('printing train and test set stats...')
    video_groups = {
        'train': stanford_train_videos,
        'test': stanford_test_videos
    }
    for group_name, video_ids in video_groups.items():
        print(group_name)
        group_video_positive_nums = [
            video_to_event_frames[video_id]['positive_frame_num']
            for video_id in video_ids
        ]
        group_video_negative_nums = [
            video_to_event_frames[video_id]['negative_frame_num']
            for video_id in video_ids
        ]
        group_video_total_nums = [
            video_to_event_frames[video_id]['total_frame_num']
            for video_id in video_ids
        ]
        print('The minimum positive frames a video contains: {}'.format(
            np.min(group_video_positive_nums)))
        print('The minimum negative frames a video contains: {}'.format(
            np.min(group_video_negative_nums)))
        print('Total positive frames: {}'.format(
            np.sum(group_video_positive_nums)))
        print('Total negative frames: {}'.format(
            np.sum(group_video_negative_nums)))


def print_okutama_person_events(annotation_dir):
    """Print car events stats in the stanford dataset.

    Args:
      annotation_dir: Annotation dir.
      video_list_file_path: List of video files to include (Default value = None).

    Returns:

    """

    annotations = io_util.load_okutama_annotation(annotation_dir)
    annotations = filter_annotation_by_label(annotations, labels=['Person'])
    video_to_event_frames = collections.defaultdict(dict)
    video_ids = list(set(annotations['videoid']))
    for video_id in video_ids:
        video_annotations = annotations[annotations['videoid'] == video_id]
        video_positive_frame_ids = list(set(video_annotations['frameid']))
        total_frame_num = okutama_video_to_frame_num[video_id]
        video_to_event_frames[video_id] = {
            'positive_frame_num':
            len(video_positive_frame_ids),
            'negative_frame_num':
            max(0, total_frame_num - len(video_positive_frame_ids)),
            'total_frame_num':
            total_frame_num,
            'positive_frame_percent':
            len(video_positive_frame_ids) * 1.0 / total_frame_num,
        }
    print('There are {} videos with person events.'.format(len(video_ids)))
    print(json.dumps(video_to_event_frames, indent=4))

    video_positive_nums = [
        video_to_event_frames[video_id]['positive_frame_num']
        for video_id in video_to_event_frames.keys()
    ]
    video_negative_nums = [
        video_to_event_frames[video_id]['negative_frame_num']
        for video_id in video_to_event_frames.keys()
    ]
    print('total positive frame num: {}'.format(np.sum(video_positive_nums)))
    print('total negative frame num: {}'.format(np.sum(video_negative_nums)))

    print('printing train and test set stats...')
    video_groups = {'train': okutama_train_videos, 'test': okutama_test_videos}
    for group_name, video_ids in video_groups.items():
        print(group_name)
        group_video_positive_nums = [
            video_to_event_frames[video_id]['positive_frame_num']
            for video_id in video_ids
        ]
        group_video_negative_nums = [
            video_to_event_frames[video_id]['negative_frame_num']
            for video_id in video_ids
        ]
        group_video_total_nums = [
            video_to_event_frames[video_id]['total_frame_num']
            for video_id in video_ids
        ]
        print('The minimum positive frames a video contains: {}'.format(
            np.min(group_video_positive_nums)))
        print('The minimum negative frames a video contains: {}'.format(
            np.min(group_video_negative_nums)))
        print('Total positive frames: {}'.format(
            np.sum(group_video_positive_nums)))
        print('Total negative frames: {}'.format(
            np.sum(group_video_negative_nums)))


def _initialize_tile_id_to_classification_label(
        tile_id_to_classification_label, grid_w, grid_h, image_id):
    # initialize all tiles to false by default
    for grid_x in range(grid_w):
        for grid_y in range(grid_h):
            tile_id = str(image_id + '_{}_{}'.format(grid_x, grid_y))
            tile_id_to_classification_label[tile_id] = False


def _get_resized_annotation(image_annotation, original_image_resolution,
                            image_resolution):
    (original_width, original_height) = original_image_resolution
    (image_width, image_height) = image_resolution
    # original annotation before resizing
    xmin, ymin, xmax, ymax = image_annotation["xmin"], \
                             image_annotation["ymin"], \
                             image_annotation["xmax"], \
                             image_annotation["ymax"]

    # convert xmin, ymin, xmax, ymax to correct values after resize
    resize_height_ratio = image_height / original_height
    resize_width_ratio = image_width / original_width
    xmin = xmin * resize_width_ratio
    xmax = xmax * resize_width_ratio
    ymin = ymin * resize_height_ratio
    ymax = ymax * resize_height_ratio
    return xmin, ymin, xmax, ymax


def get_tile_classification_annotation(
        annotation_dir, func_load_annotation_dir,
        video_id_to_original_resolution, video_id_to_frame_num, labels,
        resized_long_edge, resized_short_edge, tile_width, tile_height,
        output_dir, video_ids):
    """Get tile classificaiton annotation for resized images.

    The aspect ratio doesn't matter.
    resized_long_edge and resized_short_edge are treated as long edge and short edge.

    Args:
      annotation_dir: 
      func_load_annotation_dir: 
      video_id_to_original_resolution: 
      video_id_to_frame_num: 
      labels: 
      resized_long_edge: 
      resized_short_edge: 
      tile_width: 
      tile_height: 
      output_dir: 
      video_ids: 

    Returns:

    """
    assert video_id_to_frame_num
    assert labels
    assert resized_long_edge >= resized_short_edge

    annotations = func_load_annotation_dir(annotation_dir)
    annotations = filter_annotation_by_label(annotations, labels=labels)
    annotations = annotations[annotations['videoid'].isin(video_ids)]
    annotations['imageid'] = (
        annotations['videoid'] + '_' + annotations['frameid'].astype(str))

    print('total {} annotations'.format(len(annotations)))
    io_util.create_dir_if_not_exist(output_dir)
    for video_id in video_ids:
        original_image_resolution = (video_id_to_original_resolution[video_id])
        original_width, original_height = original_image_resolution
        if original_width >= original_height:
            resized_image_width = resized_long_edge
            resized_image_height = resized_short_edge
        else:
            resized_image_width = resized_short_edge
            resized_image_height = resized_long_edge

        grid_h, grid_w = int(resized_image_height / tile_height), int(
            resized_image_width / tile_width)
        print('{} is divided into {}x{} tiles'.format(video_id, grid_w,
                                                      grid_h))

        print('working on {}'.format(video_id))
        tile_id_to_classification_label = {}
        frame_num = video_id_to_frame_num[video_id]
        positive_num = 0
        for frame_id in range(frame_num):
            image_id = video_id + '_' + str(frame_id)
            image_annotations = annotations[annotations['imageid'] == image_id]

            _initialize_tile_id_to_classification_label(
                tile_id_to_classification_label, grid_w, grid_h, image_id)

            # debug by saving frame id = 0
            if len(image_annotations) > 0:
                im = cv2.imread('stanford/images_{}_{}/{}/{:010d}.jpg'.format(
                    resized_long_edge, resized_short_edge, video_id, frame_id +
                    1))

            for _, image_annotation in image_annotations.iterrows():
                image_resolution = (resized_image_width, resized_image_height)
                xmin, ymin, xmax, ymax = _get_resized_annotation(
                    image_annotation, original_image_resolution,
                    image_resolution)

                cv2.rectangle(im, (int(xmin), int(ymin)),
                              (int(xmax), int(ymax)), (0, 255, 0), 1)

                # upper left, upper right, lower left, lower right
                key_points = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax,
                                                                         ymax)]
                roi = (xmin, ymin, xmax, ymax)
                for (x, y) in key_points:
                    grid_x, grid_y = int(x / tile_width), int(y / tile_height)
                    grid_x = min(grid_w - 1, max(grid_x, 0))
                    grid_y = min(grid_h - 1, max(grid_y, 0))
                    tile = (grid_x * tile_width, grid_y * tile_height,
                            (grid_x + 1) * tile_width,
                            (grid_y + 1) * tile_height)
                    if is_small_bx_in_big_bx(roi, tile):
                        tile_id = str(
                            image_id + '_{}_{}'.format(grid_x, grid_y))
                        if not tile_id_to_classification_label[tile_id]:
                            tile_id_to_classification_label[tile_id] = True
                            positive_num += 1

            if len(image_annotations) > 0:
                cv2.imwrite('/tmp/{}.jpg'.format(image_id), im)

        negative_num = int(frame_num * grid_w * grid_h) - positive_num
        print(
            'For {}, number of positives: {}, number of negatives: {}'.format(
                video_id, positive_num, negative_num))
        negative_num_in_lut = len(
            [k for k, v in tile_id_to_classification_label.items() if not v])
        assert negative_num == negative_num_in_lut, ('Number of negative'
                                                     'examples do not match!')

        output_file_path = os.path.join(output_dir, '{}.pkl'.format(video_id))
        print('saving to {}'.format(output_file_path))
        with open(output_file_path, 'wb') as f:
            pickle.dump(tile_id_to_classification_label, f)


def get_okutama_tile_classification_annotation(
        annotation_dir, resized_long_edge, resized_short_edge, tile_width,
        tile_height, output_dir):
    return get_tile_classification_annotation(
        annotation_dir=annotation_dir,
        resized_long_edge=resized_long_edge,
        resized_short_edge=resized_short_edge,
        tile_width=tile_width,
        tile_height=tile_height,
        output_dir=output_dir,
        func_load_annotation_dir=io_util.load_okutama_annotation,
        video_id_to_original_resolution=(
            okutama_video_id_to_original_resolution),
        video_id_to_frame_num=okutama_video_to_frame_num,
        labels=['Person'],
        video_ids=okutama_train_videos + okutama_test_videos)


def get_stanford_tile_classification_annotation(
        annotation_dir, resized_long_edge, resized_short_edge, tile_width,
        tile_height, output_dir):
    return get_tile_classification_annotation(
        annotation_dir=annotation_dir,
        resized_long_edge=resized_long_edge,
        resized_short_edge=resized_short_edge,
        tile_width=tile_width,
        tile_height=tile_height,
        output_dir=output_dir,
        func_load_annotation_dir=io_util.load_stanford_campus_annotation,
        video_id_to_original_resolution=(
            stanford_video_id_to_original_resolution),
        video_id_to_frame_num=stanford_video_to_frame_num,
        labels=['Car', 'Bus'],
        video_ids=stanford_train_videos + stanford_test_videos)


def load_tile_classification_annotation(tile_classification_annotation_dir,
                                        video_ids=[]):
    tile_annotations = collections.defaultdict(int)
    annotation_pkl_files = glob.glob(
        os.path.join(tile_classification_annotation_dir, '*.pkl'))
    if video_ids:
        annotation_pkl_files = [
            annotation_pkl_file for annotation_pkl_file in annotation_pkl_files
            if os.path.splitext(os.path.basename(annotation_pkl_file))[0] in
            video_ids
        ]
    for annotation_pkl_file in annotation_pkl_files:
        with open(annotation_pkl_file, 'rb') as f:
            video_tile_annotations = pickle.load(f)
        tile_annotations.update(video_tile_annotations)
    return tile_annotations


if __name__ == '__main__':
    fire.Fire()
