"""Analyze stanford results."""
from __future__ import absolute_import, division, print_function

import math
import os
import collections
import json
import numpy as np

import fire
import annotation
import io_util
import redis
import matplotlib
import cPickle as pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics

from itertools import izip_longest


def plot_precision_recall_curve(y_true, probas_pred, output_file_path):
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        y_true, probas_pred)
    average_precision = sklearn.metrics.average_precision_score(
        y_true, probas_pred)
    print('average precision: {}'.format(average_precision))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    plt.savefig(output_file_path, bbox_inches='tight')


def _fix_prediction_id_to_ground_truth_id(prediction_id):
    id_splits = prediction_id.split('_')
    id_splits[-3] = str(int(id_splits[-3]) - 1)
    return '_'.join(id_splits)


def _fix_ground_truth_id_to_prediction_id(ground_truth_id):
    id_splits = ground_truth_id.split('_')
    id_splits[-3] = str(int(id_splits[-3]) + 1)
    return '_'.join(id_splits)


def analyze_prediction_metrics_on_test_images(
        gt_dir, result_dir, test_id_file_path, output_file_path):
    ground_truths = io_util.load_all_pickles_from_dir(gt_dir)
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    test_ids = pickle.load(open(test_id_file_path, 'rb'))
    positive_ids = test_ids['positive']
    negative_ids = test_ids['negative']
    print('# positive: {}, # negative: {}'.format(
        len(positive_ids), len(negative_ids)))
    total_ids = positive_ids + negative_ids
    y_true = []
    probas_pred = []
    for ground_truth_id in total_ids:
        y_true.append(ground_truths[ground_truth_id])
        # bug, frame_id for test used 1-based instead of 0-based
        prediction_id = _fix_ground_truth_id_to_prediction_id(ground_truth_id)
        probas_pred.append(predictions[prediction_id][1])
    plot_precision_recall_curve(y_true, probas_pred, output_file_path)


def analyze_prediction_metrics_on_video(gt_dir, result_dir, output_file_path):
    ground_truths = io_util.load_all_pickles_from_dir(gt_dir)
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    y_true = []
    probas_pred = []
    tp, fp, tn, fn = 0, 0, 0, 0
    for tile_id, prediction in predictions.iteritems():
        probas_pred.append(prediction[1])
        # bug, frame_id for test used 1-based instead of 0-based
        ground_truth_id = _fix_prediction_id_to_ground_truth_id(tile_id)
        assert ground_truth_id in ground_truths
        y_true.append(ground_truths[ground_truth_id])
        prediction_class = np.argmax(prediction)
        ground_truth_class = int(ground_truths[ground_truth_id])
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
    print('tp: {}, fp: {}, tn: {}, fn: {}'.format(tp, fp, tn, fn))
    recall = tp / (tp + fn * 1.0)
    precision = tp / (tp + fp * 1.0)
    fpr = fp / (fp + tn * 1.0)
    tpr = recall
    print('recall: {:.2f}, precision: {:.2f}, fpr: {:.2f}, tpr: {:.2f}'.format(
        recall, precision, fpr, tpr))
    plot_precision_recall_curve(y_true, probas_pred, output_file_path)


def analyze_event_metrics_on_video(annotation_dir, dataset_name,
                                   tile_annotation_dir, result_dir,
                                   video_list_file_path):
    dataset_to_annotation_func = {
        'okutama': io_util.load_okutama_annotation,
        'stanford_campus': io_util.load_stanford_campus_annotation,
    }
    dataset_to_labels = {
        'okutama': ['Person'],
        'stanford_campus': ['Car', 'Bus']
    }

    assert dataset_name in dataset_to_annotation_func
    assert dataset_name in dataset_to_labels

    annotations = dataset_to_annotation_func[dataset_name](annotation_dir)
    tile_annotations = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    predictions = io_util.load_all_pickles_from_dir(result_dir)

    if video_list_file_path:
        annotations = annotation.filter_annotation_by_video_name(
            annotations, video_list_file_path)
    annotations = annotation.filter_annotation_by_label(
        annotations, labels=dataset_to_labels[dataset_name])
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)

    detected_tracks = {}
    all_tracks = []
    for track_id, track_annotations in track_annotations_grp:
        all_tracks.append(track_id)
        sorted_track_annotations = track_annotations.sort_values('frameid')
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        videoid = videoid[0]
        print('video id: {}, track id: {}, start frame: {}, end frame: {}'.
              format(videoid, track_id, start_frame, end_frame))

        for index, row in sorted_track_annotations.iterrows():
            tile_detected = _has_tile_fired(tile_annotations, predictions, row)
            if tile_detected:
                detected_tracks[track_id] = row['frameid'] - start_frame
                print('first frame to be send: {}'.format(row['frameid']))
                break

    missed_tracks = set(all_tracks) - set(detected_tracks.keys())
    print('# total events: {}'.format(len(all_tracks)))
    print('# missed tracks {}, percentage {:.2f}'.format(
        len(missed_tracks),
        len(missed_tracks) * 1.0 / len(all_tracks)))
    print('Average frame latency among detected tracks: {:.2f}({:.2f})'.format(
        np.mean(detected_tracks.values()), np.std(detected_tracks.values())))
    print(('Average time latency(s) among '
           'detected tracks (assuming 30FPS): {:.2f}({:.2f})').format(
               np.mean(detected_tracks.values()) / 30.0,
               np.std(detected_tracks.values()) / 30.0))


def _get_keys_by_prefix(my_dict, key_prefix):
    keys = [key for key in my_dict.keys() if key.startswith(key_prefix)]
    return keys


def _has_tile_fired(tile_annotations, predictions, row):
    image_id = '{}_{}'.format(row['videoid'], row['frameid'])
    tile_ids = _get_keys_by_prefix(tile_annotations, image_id)
    for tile_id in tile_ids:
        prediction = np.argmax(
            predictions[_fix_ground_truth_id_to_prediction_id(tile_id)])
        if tile_annotations[tile_id] and prediction:
            return True
    return False
    # for grid_x in range(grid_w):
    #     for grid_y in range(grid_h):
    #         tile_id = str(image_id + '_{}_{}'.format(grid_x, grid_y))


if __name__ == '__main__':
    fire.Fire()
