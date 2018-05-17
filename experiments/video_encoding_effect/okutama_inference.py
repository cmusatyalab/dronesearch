from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import os
import sys

import fire
import numpy as np
# add "scripts" to pythonpath
import object_detection_evaluation
from object_detection.core import standard_fields
from PIL import Image
from logzero import logger
from tf_inference import TFModel

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts'))
import annotation_stats
import annotation

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


def _add_img_ground_truth_to_evaluator(pascal_evaluator, image_id, image_annotations):
    groundtruth_boxes = image_annotations[['ymin', 'xmin', 'ymax', 'xmax']].as_matrix().astype(np.float)
    # okutama only has 1 class
    groundtruth_class_labels = np.array([1] * groundtruth_boxes.shape[0], dtype=int)
    pascal_evaluator.add_single_ground_truth_image_info(
        image_id,
        {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes,
         standard_fields.InputDataFields.groundtruth_classes:
             groundtruth_class_labels,
         standard_fields.InputDataFields.groundtruth_difficult:
             np.array([False] * groundtruth_boxes.shape[0], dtype=bool)})


def _get_absolute_coordinate(boxes):
    """Boxes should be in ymin,xmin,ymax,xmax format (which is default by tfod api)"""
    boxes[:, 0] *= annotation_stats.okutama_original_height
    boxes[:, 2] *= annotation_stats.okutama_original_height
    boxes[:, 1] *= annotation_stats.okutama_original_width
    boxes[:, 3] *= annotation_stats.okutama_original_width
    return boxes


def _add_img_predictions_to_evaluator(pascal_evaluator, image_id, detected_boxes, detected_scores):
    detected_boxes = _get_absolute_coordinate(detected_boxes).astype(np.float)
    detected_class_labels = np.array([1] * detected_boxes.shape[0], dtype=int)
    pascal_evaluator.add_single_detected_image_info(image_id, {
        standard_fields.DetectionResultFields.detection_boxes:
            detected_boxes,
        standard_fields.DetectionResultFields.detection_scores:
            detected_scores,
        standard_fields.DetectionResultFields.detection_classes:
            detected_class_labels
    })


def auc(predictions_dir, annotation_dir):
    # okutama only detects Person
    categories = [{'id': 1, 'name': 'Person'}]
    #  Add groundtruth
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)
    annotations = annotation.load_and_filter_dataset_test_annotation('okutama', annotation_dir)
    logger.debug("calculating auc")
    for test_video in test_video_ids:
        logger.debug(test_video)
        predictions = np.load(os.path.join(predictions_dir, '{}.npy'.format(test_video)))
        image_num = annotation_stats.okutama_video_to_frame_num[test_video]
        for frame_id in range(image_num):
            image_id = '{}_{}'.format(test_video, frame_id)
            image_annotations = annotations[annotations['imageid'] == image_id]
            _add_img_ground_truth_to_evaluator(pascal_evaluator, image_id, image_annotations)
            _add_img_predictions_to_evaluator(pascal_evaluator,
                                              image_id,
                                              predictions.item().get('detection_boxes')[frame_id, :, :],
                                              predictions.item().get('detection_scores')[frame_id]
                                              )
    metrics = pascal_evaluator.evaluate()
    logger.info('AP for Person @ 0.5IOU is {}'.format(metrics['PascalBoxes_PerformanceByCategory/AP@0.5IOU/Person']))
    pascal_evaluator.clear()


def infer(frozen_graph_path, label_file_path, num_classes, image_base_dir, output_dir):
    tf_model = TFModel(frozen_graph_path, label_file_path, num_classes)
    batch_size = 10
    output_candidate_num = 100
    for test_video in test_video_ids:
        predictions = {}
        predictions['detection_boxes'] = []
        predictions['detection_scores'] = []
        # batch elements together
        image_path_iterable = _get_image_paths(os.path.join(image_base_dir, test_video))
        for image_paths in grouper(batch_size, image_path_iterable):
            image_paths = [image_path for image_path in image_paths if image_path is not None]
            logger.debug("working on {}".format('\n'.join(image_paths)))
            image_batch = _batch_load_paths(image_paths)
            result = tf_model.run_batch_inference(image_batch)
            # image number should be the same
            assert result['detection_scores'].shape[0] == result['detection_boxes'].shape[0] == image_batch.shape[0]
            # the tf model is configured to output at maxmum 100 prediction per class
            predictions['detection_boxes'].extend(result['detection_boxes'][:, :output_candidate_num, :])
            predictions['detection_scores'].extend(result['detection_scores'][:, :output_candidate_num])
        output_path = os.path.join(output_dir, '{}.npy'.format(test_video))
        predictions['detection_boxes'] = np.array(predictions['detection_boxes'])
        predictions['detection_scores'] = np.array(predictions['detection_scores'])
        with open(output_path, 'wb') as f:
            np.save(f, predictions)
        logger.info("Finished writing to {}".format(output_path))


if __name__ == '__main__':
    fire.Fire()
