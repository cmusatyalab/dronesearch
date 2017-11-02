"""Annotation Wrapper and related logic.
"""
from __future__ import (absolute_import, division, print_function,
    unicode_literals)

import abc
import glob
import os

import io_util


def get_positive_annotation_mask(annotations):
    """Car/NoCar annotation pandas dataframe mask."""
    lost_mask = (annotations['lost'] == 0)
    label_mask = annotations['label'].isin(['Car', 'Bus'])
    mask = label_mask & lost_mask
    return mask


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
        return os.path.join(self._slice_image_dir,
                            '*{}*'.format(imageid))


class StanfordCarSliceAnnotations(SliceAnnotations):
    """Annotations for car/nocar classification on Stanford dataset.

    The annotation_dir for stanford dataset loads all annotations. Filtering
    are performed to return only the imageids and annotations of interests.

    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # annotations for full resolution images
        self._base_annotations = io_util.load_stanford_campus_annotation(
            self._annotation_dir)
        self._imageids = None
        self._annotations = None

    @property
    def imageids(self):
        if not self._imageids:
            slice_file_paths = glob.glob(
                os.path.join(self._slice_image_dir, '*'))
            slice_file_basenames = [
                os.path.basename(file_path) for file_path in slice_file_paths]
            file_basename_components = [file_basename.split('_') for
                                        file_basename in slice_file_basenames]
            imageids = ['_'.join([components[0], components[1],
                                  str(int(components[3]))]) for components in
                        file_basename_components]
            self._imageids = list(set(imageids))
        return self._imageids

    @property
    def annotations(self):
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

    def get_sliced_images_path_pattern(self, imageid):
        components = imageid.split('_')
        video_id = '_'.join([components[0], components[1]])
        frame_id = components[2]
        return os.path.join(self._slice_image_dir,
                            '*{}_video.mov_*0{}_*'.format(video_id,
                                                          frame_id))
