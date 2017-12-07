"""Common Plot Functionalities."""
from __future__ import absolute_import, division, print_function

import cPickle as pickle
import cv2
import glob
import json
import collections
import math
import numpy as np
import os

import annotation
import annotation_stats
import fire
import io_util
import result_analysis

import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "font.size": 25
}
mpl.rcParams.update(pgf_with_rc_fonts)

import logzero
import sklearn
from logzero import logger
logzero.logfile("plot_util.log", maxBytes=1e6, backupCount=3, mode='a')

dir_path = os.path.dirname(os.path.realpath(__file__))
experiments = {
    'interval_sampling': {
        dataset_name: os.path.join(
            dir_path, dataset_name,
            'experiments/classification_448_224_224_224_extra_negative/interval'
        )
        for dataset_name in result_analysis.datasets.keys()
    },
    'prc': {
        dataset_name: os.path.join(
            dir_path, dataset_name,
            'experiments/classification_448_224_224_224_extra_negative/test_inference_proba'
        )
        for dataset_name in result_analysis.datasets.keys()
    }
}


def _fix_prediction_id_to_ground_truth_id(prediction_id):
    """

    Args:
      prediction_id: 

    Returns:

    """
    prediction_id = os.path.basename(prediction_id)
    id_splits = prediction_id.split('_')
    id_splits[-3] = str(int(id_splits[-3]) - 1)
    return '_'.join(id_splits)


def get_confusion_matrix_from_prediction_and_ground_truth(
        prediction_classes, ground_truths):
    tp, fp, fn, tn = 0, 0, 0, 0
    for tile_id, prediction_class in prediction_classes.iteritems():
        gt_id = _fix_prediction_id_to_ground_truth_id(tile_id)
        ground_truth_class = ground_truths[gt_id]
        if (prediction_class == 1) and (
                prediction_class == ground_truth_class):
            tp += 1
        elif (prediction_class == 0) and (
                prediction_class == ground_truth_class):
            tn += 1
        elif (prediction_class == 1) and (prediction_class !=
                                          ground_truth_class):
            fp += 1
        elif (prediction_class == 0) and (prediction_class !=
                                          ground_truth_class):
            fn += 1
    total_num = tp + tn + fp + fn
    print('tp: {}, fp: {}, tn: {}, fn: {}. total_num: {}'.format(
        tp, fp, tn, fn, total_num))
    return tp, fp, tn, fn


def get_precision_recall_from_confusion_matrix(confusion_matrix):
    assert len(confusion_matrix) == 4
    tp, fp, tn, fn = confusion_matrix
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall


def get_ground_truth(dataset_name):
    tile_annotation_dir = os.path.join(
        dir_path, result_analysis.datasets[dataset_name][1])
    ground_truth = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    ground_truth = collections.defaultdict(bool, ground_truth)
    return ground_truth


def get_predictions(experiment_name, dataset_name):
    result_dir = experiments[experiment_name][dataset_name]
    return io_util.load_all_pickles_from_dir(result_dir)


def get_precision_recall_curve_from_prediction_and_ground_truth(
        predictions, ground_truths):
    y_true, probas_pred = [], []
    for tile_id, prediction in predictions.iteritems():
        probas_pred.append(prediction[1])
        gt_id = _fix_prediction_id_to_ground_truth_id(tile_id)
        y_true.append(ground_truths[gt_id])
    return _get_precision_recall_curve(y_true, probas_pred)


def _get_precision_recall_curve(y_true, probas_pred):
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        y_true, probas_pred)
    average_precision = sklearn.metrics.average_precision_score(
        y_true, probas_pred)
    return precision, recall, threshold, average_precision
    # print('average precision: {}'.format(average_precision))
    # plt.step(recall, precision, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
    #     average_precision))
    # plt.savefig(output_file_path, bbox_inches='tight')


def get_event_accuracy_random_drop(ground_truth):
    pass
