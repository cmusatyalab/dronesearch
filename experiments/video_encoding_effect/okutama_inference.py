from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import os
import sys

import fire
import numpy as np
from PIL import Image
from logzero import logger
from tf_inference import TFModel

# add "scripts" to pythonpath
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
import annotation_stats

test_video_ids = annotation_stats.okutama_test_videos


def _get_image_paths(image_dir):
    """
    Return image paths from the base directory
    :param image_base_dir: a directory that contains a list of directories that corresponds to each video
    :return:
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    for image_path in image_paths:
        yield image_path


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def _batch_load_paths(paths):
    image_batch = []
    for path in paths:
        if path is not None:
            image = Image.open(path)
            image_batch.append(np.array(image))
    return np.array(image_batch)


def main(frozen_graph_path, label_file_path, num_classes, image_base_dir, output_dir):
    tf_model = TFModel(frozen_graph_path, label_file_path, num_classes)
    batch_size = 10
    output_candidate_num = 100
    for test_video in test_video_ids:
        # batch elements together
        predictions = []
        image_path_iterable = _get_image_paths(os.path.join(image_base_dir, test_video))
        for image_paths in grouper(batch_size, image_path_iterable):
            logger.debug("working on {}".format('\n'.join(image_paths)))
            image_batch = _batch_load_paths(image_paths)
            result = tf_model.run_batch_inference(image_batch)
            # image number should be the same
            assert result['detection_scores'].shape[0] == result['detection_boxes'].shape[0] == image_batch.shape[0]
            result['detection_scores'] = np.expand_dims(result['detection_scores'], axis=2)
            result = np.concatenate((result['detection_boxes'], result['detection_scores']), axis=2)
            # remove zero confidence boxes
            result = result[result[:, :, 4] > 0.0]
            result = result.reshape((batch_size, output_candidate_num, 5))
            predictions.extend(result)
        output_path = os.path.join(output_dir, '{}.npy'.format(test_video))
        with open(output_path, 'wb') as f:
            np.save(f, predictions)
        logger.info("Finished writing to {}".format())


if __name__ == '__main__':
    fire.Fire(main)
