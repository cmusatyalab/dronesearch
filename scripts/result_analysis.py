"""Analyze stanford results."""
from __future__ import absolute_import, division, print_function

import cPickle as pickle
import collections
import cv2
import glob
import itertools
import functools
import json
import math
import numpy as np
import os
from itertools import izip_longest

import annotation
import annotation_stats
import fire
import logzero
import io_util
import redis
import sklearn.metrics
from logzero import logger

logzero.logfile("result_analysis.log", maxBytes=1e6, backupCount=3, mode='a')
dir_path = os.path.dirname(os.path.realpath(__file__))


def _fix_prediction_id_to_ground_truth_id(prediction_id):
    id_splits = prediction_id.split('_')
    id_splits[-3] = str(int(id_splits[-3]) - 1)
    return '_'.join(id_splits)


def _fix_ground_truth_id_to_prediction_id(ground_truth_id,
                                          prediciton_id_prefix):
    id_splits = ground_truth_id.split('_')
    id_splits[-3] = str(int(id_splits[-3]) + 1)
    return prediciton_id_prefix + '_'.join(id_splits)


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


def analyze_prediction_metrics_on_video(gt_dir,
                                        result_dir,
                                        plot=False,
                                        output_file_path=None):
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
    print('positives: {}, negatives: {}, total: {}'.format(
        (tp + fn), (fp + tn), (tp + fn + fp + tn)))
    recall = tp / (tp + fn * 1.0)
    precision = tp / (tp + fp * 1.0)
    fpr = fp / (fp + tn * 1.0)
    tpr = recall
    print('recall: {:.2f}, precision: {:.2f}, fpr: {:.2f}, tpr: {:.2f}'.format(
        recall, precision, fpr, tpr))

    # get precision recall curve
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(
        y_true, probas_pred)
    average_precision = sklearn.metrics.average_precision_score(
        y_true, probas_pred)

    if plot:
        plot_precision_recall_curve(y_true, probas_pred, output_file_path)
    else:
        return precision, recall, threshold, average_precision


def all_prediction_metrics_on_video(output_dir):
    datasets = {
        'Elephant':
        ('elephant/classification_448_224_224_224_annotations',
         'elephant/experiments/classification_448_224_224_224/test_inference'),
        'Raft':
        ('raft/classification_448_224_224_224_annotations',
         'raft/experiments/classification_448_224_224_224/test_inference'),
        'Human':
        ('okutama/classification_1792_1792_annotations',
         'okutama/experiments/classification_1792_1792/test_inference'),
        'Car':
        ('stanford/classification_448_224_224_224_annotations',
         'stanford/experiments/classification_448_224_224_224/test_inference'),
    }
    dataset_to_precision_recall = {}
    for dataset_name, (gt_dir, result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        precision, recall, threshold, average_precision = analyze_prediction_metrics_on_video(
            gt_dir, result_dir)
        dataset_to_precision_recall[dataset_name] = {
            'precision': precision,
            'recall': recall,
            'threshold': threshold,
            'average_precision': average_precision
        }
    io_util.create_dir_if_not_exist(output_dir)
    with open(os.path.join(output_dir, 'precision_recall.pkl'), 'wb') as f:
        pickle.dump(dataset_to_precision_recall, f)


def fix_okutama_annotataions():
    dataset_name = 'okutama'
    annotation_dir = datasets[dataset_name][0]
    load_annotation_func = annotation_stats.dataset[dataset_name][
        'annotation_func']
    labels = annotation_stats.dataset[dataset_name]['labels']

    annotations = load_annotation_func(annotation_dir)
    test_video_ids = annotation_stats.dataset[dataset_name]['test']
    annotations = annotations[annotations['videoid'].isin(test_video_ids)]

    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)

    all_tracks = {}
    for track_id, track_annotations in track_annotations_grp:
        sorted_track_annotations = track_annotations.sort_values('frameid')
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        if len(sorted_track_annotations) < 10:
            print('{} is less than 10 frames, only have {} frames!'.format(
                track_id, len(sorted_track_annotations)))
            continue

        start_box = sorted_track_annotations[sorted_track_annotations[
            'frameid'] == start_frame][['xmin', 'ymin', 'xmax',
                                        'ymax']].as_matrix()
        end_box = sorted_track_annotations[sorted_track_annotations[
            'frameid'] == end_frame][['xmin', 'ymin', 'xmax',
                                      'ymax']].as_matrix()
        assert (start_box).shape == (1, 4)
        assert (end_box).shape == (1, 4)
        start_box = np.squeeze(start_box)
        end_box = np.squeeze(end_box)
        all_tracks[track_id] = (start_frame, end_frame, start_box, end_box)

    for t1, (t1_start, t1_end, t1_start_box,
             t1_end_box) in all_tracks.iteritems():
        print('searching duplicates for {}'.format(t1))
        video_tracks = _get_keys_by_id_prefix(all_tracks, t1.split('_')[0])
        for t2, (t2_start, t2_end, t2_start_box,
                 t2_end_box) in all_tracks.iteritems():
            # need to be same video
            if t2 not in video_tracks:
                continue
            if (t1 != t2) and ((t1_end == t2_start) or
                               (t1_end == t2_start - 1)):
                if annotation.iou(t1_end_box, t2_start_box) > 0.5:
                    print(t1, t2)


def analyze_event_metrics_on_video(
        annotation_dir,
        dataset_name,
        tile_annotation_dir,
        result_dir,
        threshold,  # A prediction >=threshold would be considered positive
        result_pkl_file_list,
        test_only=True):
    assert dataset_name in annotation_stats.dataset.keys()

    load_annotation_func = annotation_stats.dataset[dataset_name][
        'annotation_func']
    labels = annotation_stats.dataset[dataset_name]['labels']

    annotations = load_annotation_func(annotation_dir)
    tile_annotations = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    predictions = io_util.load_all_pickles_from_dir(
        result_dir, video_ids=result_pkl_file_list)
    if test_only:
        test_video_ids = annotation_stats.dataset[dataset_name]['test']
        annotations = annotations[annotations['videoid'].isin(test_video_ids)]
    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)

    detected_tracks = {}
    all_tracks = []
    track_length = []
    for track_id, track_annotations in track_annotations_grp:
        sorted_track_annotations = track_annotations.sort_values('frameid')
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        if len(sorted_track_annotations) < 10:
            # print('{} is less than 10 frames!'.format(track_id))
            continue

        all_tracks.append(track_id)
        videoid = videoid[0]
        print('video id: {}, track id: {}, start frame: {}, end frame: {}'.
              format(videoid, track_id, start_frame, end_frame))
        track_length.append(end_frame - start_frame + 1)

        for index, row in sorted_track_annotations.iterrows():
            tile_detected = _has_tile_fired(
                tile_annotations,
                predictions,
                row,
                prediction_id_prefix=dataset_name + '/',
                threshold=threshold)
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
    track_length.sort()
    print('track length: {}'.format(track_length))


def _get_keys_by_id_prefix(my_dict, key_prefix, default_prefix_separator='_'):
    key_prefix += default_prefix_separator
    keys = [key for key in my_dict.iterkeys() if key.startswith(key_prefix)]
    return keys


def _get_tile_resolution_from_image_resolution(
        image_resolution, long_edge_ratio, short_edge_ratio):
    im_w, im_h = image_resolution
    if im_h > im_w:
        tile_height = int(im_h * long_edge_ratio)
        tile_width = int(im_w * short_edge_ratio)
    else:
        tile_width = int(im_w * long_edge_ratio)
        tile_height = int(im_h * short_edge_ratio)
    return tile_width, tile_height


def _get_tile_coords_from_bbox(image_resolution, bbox, long_edge_ratio,
                               short_edge_ratio):
    im_w, im_h = image_resolution
    tile_width, tile_height = _get_tile_resolution_from_image_resolution(
        image_resolution, long_edge_ratio, short_edge_ratio)
    xmin, ymin, xmax, ymax = bbox
    tile_id_x, tile_id_y = [], []
    for point, length in [(xmin, tile_width), (xmax, tile_width)]:
        tile_id = int(point / length)
        tile_id_x.append(tile_id)

    for point, length in [(ymin, tile_height), (ymax, tile_height)]:
        tile_id = int(point / length)
        tile_id_y.append(tile_id)

    tile_coords = set(zip(tile_id_x, tile_id_y))
    return list(tile_coords)


def _clamp_bbox(image_resolution, bbox):
    im_w, im_h = image_resolution
    xmin, ymin, xmax, ymax = bbox
    xmin, xmax = map(lambda x: min(max(0, x), im_w - 1), [xmin, xmax])
    ymin, ymax = map(lambda x: min(max(0, x), im_h - 1), [ymin, ymax])
    return xmin, ymin, xmax, ymax


def _get_positive_tile_proba(image_resolution,
                             predictions,
                             row,
                             long_edge_ratio=0.5,
                             short_edge_ratio=1,
                             prediction_id_prefix=''):
    image_width, image_height = image_resolution
    bbox = _clamp_bbox(image_resolution,
                       (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
    tile_coords = _get_tile_coords_from_bbox(image_resolution, bbox,
                                             long_edge_ratio, short_edge_ratio)
    tile_fire_thresholds = []
    for tile_coord in tile_coords:
        tile_id = '{}_{}_{}_{}'.format(row['videoid'], row['frameid'],
                                       *tile_coord)
        if _fix_ground_truth_id_to_prediction_id(
                tile_id, prediction_id_prefix) not in predictions:
            # continue
            import pdb
            pdb.set_trace()

        # commented out for suppressed predictions
        # assert _fix_ground_truth_id_to_prediction_id(
        #     tile_id, prediction_id_prefix) in predictions
        pred_prob = predictions[_fix_ground_truth_id_to_prediction_id(
            tile_id, prediction_id_prefix)][1]
        tile_fire_thresholds.append(pred_prob)
    return tile_fire_thresholds


def _has_tile_fired(tile_annotations,
                    predictions,
                    row,
                    prediction_id_prefix,
                    threshold=0.5):
    image_id = '{}_{}'.format(row['videoid'], row['frameid'])
    tile_ids = _get_keys_by_id_prefix(tile_annotations, image_id)
    for tile_id in tile_ids:
        prediction = predictions[_fix_ground_truth_id_to_prediction_id(
            tile_id, prediction_id_prefix)][1] >= threshold
        if tile_annotations[tile_id] and prediction:
            return True
    return False
    # for grid_x in range(grid_w):
    #     for grid_y in range(grid_h):
    #         tile_id = str(image_id + '_{}_{}'.format(grid_x, grid_y))


def _get_tile_coord_from_tile_id(tile_id):
    return '_'.join(tile_id.split('_')[-2:])


def _get_tile(image_id, image_dir, long_edge_ratio, short_edge_ratio):
    """Assumes image_id (tile_id) is 1-based. Matches prediction keys.

    Args:
      image_id: tile_id
      image_dir: 
      long_edge_ratio: 
      short_edge_ratio: 

    Returns:

    """

    assert os.path.exists(image_dir)

    image_id_contents = image_id.split('_')
    if len(image_id_contents) == 5:
        # print(('I guessed you are using Stanford dataset. '
        #        'If not, please double-check'))
        video_id = '_'.join([image_id_contents[0], image_id_contents[1]])
        (frame_id, grid_x,
         grid_y) = (image_id_contents[2], image_id_contents[3],
                    image_id_contents[4])
    elif len(image_id_contents) == 4:
        # print(('I guessed you are using Okutama/elephant/raft dataset. '
        #        'If not, please double-check'))
        video_id, frame_id, grid_x, grid_y = (image_id_contents[0],
                                              image_id_contents[1],
                                              image_id_contents[2],
                                              image_id_contents[3])
    else:
        raise ValueError(
            'Not recognized image_id {} from annotations.'.format(image_id))

    frame_id = int(frame_id)
    grid_x = int(grid_x)
    grid_y = int(grid_y)
    base_image_path = os.path.join(image_dir, video_id, '{:010d}'.format(
        int(frame_id))) + '.jpg'
    im = cv2.imread(base_image_path)
    if im is None:
        raise ValueError('Failed to load image: {}'.format(base_image_path))
    im_h, im_w, _ = im.shape
    if im_h > im_w:
        tile_height = int(im_h * long_edge_ratio)
        tile_width = int(im_w * short_edge_ratio)
    else:
        tile_width = int(im_w * long_edge_ratio)
        tile_height = int(im_h * short_edge_ratio)

    tile_x = grid_x * tile_width
    tile_y = grid_y * tile_height
    current_tile = im[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
    ret, encoded_tile = cv2.imencode('.jpg', current_tile)
    if not ret:
        raise ValueError('Failed to encode tile: '.format(image_id))
    return encoded_tile.tobytes()


def generate_positive_streams(dataset_name,
                              result_dir,
                              output_dir,
                              image_dir,
                              long_edge_ratio=0.5,
                              short_edge_ratio=1):
    video_id_to_frame_num = annotation_stats.dataset[dataset_name][
        'video_id_to_frame_num']
    test_video_ids = annotation_stats.dataset[dataset_name]['test']
    predictions = io_util.load_all_pickles_from_dir(result_dir)

    for video_id in test_video_ids:
        print('working on {}'.format(video_id))
        video_predictions = _get_keys_by_id_prefix(predictions, video_id)
        assert video_predictions

        tile_coord_sets = set(
            [_get_tile_coord_from_tile_id(key) for key in video_predictions])
        output_video_dir = os.path.join(output_dir, video_id)
        for tile_coord in tile_coord_sets:
            io_util.create_dir_if_not_exist(
                os.path.join(output_video_dir, tile_coord))
        stream_dir_to_frame_id = {
            tile_coord: 0
            for tile_coord in tile_coord_sets
        }

        frame_num = video_id_to_frame_num[video_id]
        for frame_id in range(frame_num):
            image_id = '{}_{}'.format(video_id, frame_id)

            # fix prediciton 1-based
            prediction_image_id = '{}_{}'.format(video_id, frame_id + 1)
            prediction_tile_ids = _get_keys_by_id_prefix(
                predictions, prediction_image_id)
            for prediction_tile_id in prediction_tile_ids:
                prediction = np.argmax(predictions[prediction_tile_id])
                if prediction:
                    tile_im = _get_tile(prediction_tile_id, image_dir,
                                        long_edge_ratio, short_edge_ratio)
                    tile_coord = _get_tile_coord_from_tile_id(
                        prediction_tile_id)
                    output_frame_id = stream_dir_to_frame_id[tile_coord]
                    stream_dir_to_frame_id[tile_coord] += 1
                    output_tile_path = os.path.join(
                        output_video_dir, tile_coord,
                        '{:010d}.jpg'.format(output_frame_id))
                    with open(output_tile_path, 'wb') as f:
                        f.write(tile_im)


# unsuppressed use 'test_inference_proba'
# suppressed use 'suppress'
datasets = {
    'elephant':
    ('elephant/annotations',
     'elephant/classification_448_224_224_224_annotations',
     'elephant/experiments/classification_448_224_224_224_extra_negative/test_inference_proba'
     ),
    'raft':
    ('raft/annotations', 'raft/classification_448_224_224_224_annotations',
     'raft/experiments/classification_448_224_224_224_extra_negative/test_inference_proba'
     ),
    'okutama':
    ('okutama/annotations',
     'okutama/classification_448_224_224_224_annotations',
     'okutama/experiments/classification_448_224_224_224_extra_negative/test_inference_proba'
     ),
    'stanford':
    ('stanford/annotations',
     'stanford/classification_448_224_224_224_annotations',
     'stanford/experiments/classification_448_224_224_224_extra_negative/test_inference_proba'
     ),
}


def all_extract_prediction_probability():
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        pkl_files = glob.glob(os.path.join(result_dir, '*.pkl'))
        output_dir = os.path.join(
            os.path.dirname(result_dir), 'test_inference_proba')
        io_util.create_dir_if_not_exist(output_dir)
        for pkl_file in pkl_files:
            prediction_dict = pickle.load(open(pkl_file, 'rb'))
            prediction_proba_dict = {
                k: v[:2]
                for k, v in prediction_dict.iteritems()
            }
            output_file = os.path.join(output_dir, os.path.basename(pkl_file))
            pickle.dump(prediction_proba_dict, open(output_file, 'wb'))


def all_positive_streams(output_dir):
    for dataset_name, (image_dir, result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        image_dir = os.path.join(dataset_name, 'images')
        generate_positive_streams(
            dataset_name,
            result_dir,
            dataset_output_dir,
            image_dir,
            long_edge_ratio=0.5,
            short_edge_ratio=1)


def all_extract_prediction_positives(threshold, output_dir):
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        print('workign on {}'.format(dataset_name))
        dataset_output_dir = os.path.join(
            os.path.join(output_dir), dataset_name)
        io_util.create_dir_if_not_exist(dataset_output_dir)
        extract_prediction_positives(result_dir, dataset_output_dir, threshold)


def _sample_image_from_extra_negative_prediction(
        prediction_id, long_edge_ratio, short_edge_ratio):
    dataset_name = os.path.dirname(prediction_id)
    image_dir = os.path.join(dataset_name, 'images')
    assert os.path.exists(image_dir)
    prediction_id = os.path.basename(prediction_id)
    return _get_tile(prediction_id, image_dir, long_edge_ratio,
                     short_edge_ratio)


def extract_prediction_positives(result_dir,
                                 output_dir,
                                 threshold,
                                 long_edge_ratio=0.5,
                                 short_edge_ratio=1):
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    print('total prediction num: {}'.format(len(predictions)))
    pos_num = 0
    for prediction_id, prediction_value in predictions.iteritems():
        prediction_proba = prediction_value[1]
        if prediction_proba >= threshold:
            pos_num += 1
            # tile_im = _sample_image_from_extra_negative_prediction(
            #     prediction_id, long_edge_ratio, short_edge_ratio)
            # output_tile_path = os.path.join(
            #     output_dir, prediction_id.replace('/', '_')) + '.jpg'
            # with open(output_tile_path, 'wb') as f:
            #     f.write(tile_im)
    print('predicted positive num: {}, percentage: {}'.format(
        pos_num, pos_num / len(predictions)))


# datasets = {
#     'elephant':
#     ('elephant/annotations',
#      'elephant/classification_448_224_224_224_annotations',
#      'elephant/experiments/classification_448_224_224_224/test_inference'),
#     'raft': ('raft/annotations',
#              'raft/classification_448_224_224_224_annotations',
#              'raft/experiments/classification_448_224_224_224/test_inference'),
#     'okutama': ('okutama/annotations',
#                 'okutama/classification_1792_1792_annotations',
#                 'okutama/experiments/classification_1792_1792/test_inference'),
#     'stanford':
#     ('stanford/annotations',
#      'stanford/classification_448_224_224_224_annotations',
#      'stanford/experiments/classification_448_224_224_224/test_inference'),
# }


def all_event_metrics_on_video(threshold):
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        if dataset_name == 'stanford':
            result_pkl_file_list = [
                '{}_test_inference_results_horizontal'.format(dataset_name),
                '{}_test_inference_results_vertical'.format(dataset_name),
            ]
        else:
            result_pkl_file_list = [
                '{}_test_inference_results'.format(dataset_name)
            ]
        analyze_event_metrics_on_video(
            annotation_dir,
            dataset_name,
            tile_annotation_dir,
            result_dir,
            threshold,
            result_pkl_file_list,
            test_only=True)


def all_sample_event_recall_on_video(output_dir, interval=5000):
    dataset_to_event_recall = {}
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        track_to_fire_thresholds, predictions_thresholds, ground_truth = analyze_sample_event_recall_on_video(
            annotation_dir,
            dataset_name,
            result_dir,
            tile_annotation_dir,
            long_edge_ratio=0.5,
            short_edge_ratio=1,
            interval=interval)
        dataset_to_event_recall[dataset_name] = {}
        dataset_to_event_recall[dataset_name][
            'track_to_fire_thresholds'] = track_to_fire_thresholds
        dataset_to_event_recall[dataset_name][
            'predictions_thresholds'] = predictions_thresholds
        dataset_to_event_recall[dataset_name]['ground_truth'] = ground_truth
    io_util.create_dir_if_not_exist(output_dir)
    with open(
            os.path.join(output_dir,
                         'sample_event_recall_{}.pkl'.format(interval)),
            'wb') as f:
        pickle.dump(dataset_to_event_recall, f)


def all_event_recall_on_video(output_dir):
    dataset_to_event_recall = {}
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        print('working on {}'.format(dataset_name))
        track_to_fire_thresholds, predictions_thresholds, ground_truth = analyze_event_recall_on_video(
            annotation_dir,
            dataset_name,
            result_dir,
            tile_annotation_dir,
            long_edge_ratio=0.5,
            short_edge_ratio=1)
        dataset_to_event_recall[dataset_name] = {}
        dataset_to_event_recall[dataset_name][
            'track_to_fire_thresholds'] = track_to_fire_thresholds
        dataset_to_event_recall[dataset_name][
            'predictions_thresholds'] = predictions_thresholds
        dataset_to_event_recall[dataset_name]['ground_truth'] = ground_truth
    io_util.create_dir_if_not_exist(output_dir)
    with open(os.path.join(output_dir, 'event_recall.pkl'), 'wb') as f:
        pickle.dump(dataset_to_event_recall, f)


def _filter_by_video_ids(predictions, video_ids):
    video_prediction_keys = []
    for video_id in video_ids:
        video_prediction_keys.extend(
            _get_keys_by_id_prefix(predictions, video_id))
    assert len(video_prediction_keys) <= len(predictions.keys())
    predictions = {
        k: v
        for k, v in predictions.iteritems() if k in video_prediction_keys
    }
    return predictions


def analyze_sample_event_recall_on_video(annotation_dir,
                                         dataset_name,
                                         result_dir,
                                         tile_annotation_dir,
                                         long_edge_ratio=0.5,
                                         short_edge_ratio=1,
                                         interval=100):

    assert dataset_name in annotation_stats.dataset.keys()

    load_annotation_func = annotation_stats.dataset[dataset_name][
        'annotation_func']
    labels = annotation_stats.dataset[dataset_name]['labels']
    video_id_to_original_resolution = annotation_stats.dataset[dataset_name][
        'video_id_to_original_resolution']

    tile_annotations = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    annotations = load_annotation_func(annotation_dir)
    predictions = io_util.load_all_pickles_from_dir(result_dir)

    test_video_ids = annotation_stats.dataset[dataset_name]['test']
    annotations = annotations[annotations['videoid'].isin(test_video_ids)]

    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)

    track_to_fire_thresholds = collections.defaultdict(list)

    for track_id, track_annotations in track_annotations_grp:
        sorted_track_annotations = track_annotations.sort_values('frameid')
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        if len(sorted_track_annotations) < 10:
            print('{} is less than 10 frames!'.format(track_id))
            continue

        videoid = videoid[0]
        print('video id: {}, track id: {}, start frame: {}, end frame: {}'.
              format(videoid, track_id, start_frame, end_frame))

        image_resolution = video_id_to_original_resolution[videoid]
        track_to_fire_thresholds[track_id] = []
        for index, row in sorted_track_annotations.iterrows():
            frame_id = row['frameid']

            if frame_id % interval == 0:
                tile_fire_thresholds = _get_positive_tile_proba(
                    image_resolution,
                    predictions,
                    row,
                    long_edge_ratio,
                    short_edge_ratio,
                    prediction_id_prefix=dataset_name + '/')
                track_to_fire_thresholds[track_id].extend(tile_fire_thresholds)

    predictions_thresholds = {}
    for k, v in predictions.iteritems():
        if int(os.path.basename(k).split('_')[-3]) % interval == 0:
            predictions_thresholds[k] = v[1]
    return track_to_fire_thresholds, predictions_thresholds, tile_annotations


def analyze_event_recall_on_video(annotation_dir,
                                  dataset_name,
                                  result_dir,
                                  tile_annotation_dir,
                                  test_only=True,
                                  long_edge_ratio=0.5,
                                  short_edge_ratio=1):
    assert dataset_name in annotation_stats.dataset.keys()

    load_annotation_func = annotation_stats.dataset[dataset_name][
        'annotation_func']
    labels = annotation_stats.dataset[dataset_name]['labels']
    video_id_to_original_resolution = annotation_stats.dataset[dataset_name][
        'video_id_to_original_resolution']

    tile_annotations = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    annotations = load_annotation_func(annotation_dir)
    predictions = io_util.load_all_pickles_from_dir(result_dir)

    if test_only:
        test_video_ids = annotation_stats.dataset[dataset_name]['test']
        annotations = annotations[annotations['videoid'].isin(test_video_ids)]
        # comment out to assume the prediction pickle dir only has results for test videos
        # predictions = _filter_by_video_ids(predictions, test_video_ids)

    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)

    track_to_fire_thresholds = collections.defaultdict(list)
    for track_id, track_annotations in track_annotations_grp:
        sorted_track_annotations = track_annotations.sort_values('frameid')
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        if len(sorted_track_annotations) < 10:
            print('{} is less than 10 frames!'.format(track_id))
            continue

        videoid = videoid[0]
        print('video id: {}, track id: {}, start frame: {}, end frame: {}'.
              format(videoid, track_id, start_frame, end_frame))

        image_resolution = video_id_to_original_resolution[videoid]
        for index, row in sorted_track_annotations.iterrows():
            tile_fire_thresholds = _get_positive_tile_proba(
                image_resolution,
                predictions,
                row,
                long_edge_ratio,
                short_edge_ratio,
                prediction_id_prefix=dataset_name + '/')
            track_to_fire_thresholds[track_id].extend(tile_fire_thresholds)
    predictions_thresholds = {k: v[1] for k, v in predictions.items()}
    return track_to_fire_thresholds, predictions_thresholds, tile_annotations


def random_select_accuracy(dataset_name, random_sample_intervals=[]):
    annotation_dir = os.path.join(dir_path, datasets[dataset_name][0])
    annotations = annotation.load_and_filter_dataset_test_annotation(
        dataset_name, annotation_dir)
    random_select_results = collections.OrderedDict()
    for random_sample_interval in random_sample_intervals:
        track_iter = annotation.get_track_iter(annotations)
        all_event_num = 0
        event_detected_num = 0
        for track_id, track_annotations in track_iter:
            logger.debug('working on {}'.format(track_id))
            all_event_num += 1
            sorted_track_annotations = track_annotations.sort_values('frameid')
            videoid = list(set(sorted_track_annotations['videoid']))
            assert len(videoid) == 1
            videoid = videoid[0]
            detected = False
            for index, row in sorted_track_annotations.iterrows():
                if row['frameid'] % random_sample_interval == 0:
                    detected = True
                    break
            if detected:
                event_detected_num += 1
        event_recall = event_detected_num / all_event_num
        random_select_results[random_sample_interval] = event_recall
    return random_select_results


def get_event_lengths(dataset_name):
    annotation_dir = os.path.join(dir_path, datasets[dataset_name][0])
    annotations = annotation.load_and_filter_dataset_test_annotation(
        dataset_name, annotation_dir)
    track_iter = annotation.get_track_iter(annotations)
    event_lengths = []
    for track_id, track_annotations in track_iter:
        logger.debug('working on {}'.format(track_id))
        sorted_track_annotations = track_annotations.sort_values('frameid')
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        videoid = videoid[0]
        if len(sorted_track_annotations) < 10:
            logger.info(
                '{} {} is less than 10 frames, only have {} frames!'.format(
                    dataset_name, track_id, len(sorted_track_annotations)))
        event_lengths.append(len(sorted_track_annotations))
    return event_lengths


def get_okutama_action_prediction_stats(output_dir,
                                        action='Walking',
                                        long_edge_ratio=0.5,
                                        short_edge_ratio=1):
    dataset_name = 'okutama'
    annotation_dir = 'okutama/annotations'
    annotations = annotation.load_and_filter_dataset_test_annotation(
        dataset_name, annotation_dir)
    annotations = annotations[annotations['action'] == action]
    track_iter = annotation.get_track_iter(annotations)

    result_dir = datasets[dataset_name][2]
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    video_id_to_original_resolution = annotation_stats.dataset[dataset_name][
        'video_id_to_original_resolution']

    track_to_fire_thresholds = collections.defaultdict(list)
    track_to_event_interval = collections.defaultdict(tuple)
    for track_id, track_annotations in track_iter:
        logger.debug('working on {}'.format(track_id))
        sorted_track_annotations = track_annotations.sort_values('frameid')
        videoid = list(set(sorted_track_annotations['videoid']))
        assert len(videoid) == 1
        videoid = videoid[0]
        image_resolution = video_id_to_original_resolution[videoid]
        start_frame = min(sorted_track_annotations['frameid'])
        end_frame = max(sorted_track_annotations['frameid'])
        track_to_event_interval[track_id] = (start_frame, end_frame)
        for index, row in sorted_track_annotations.iterrows():
            tile_fire_thresholds = _get_positive_tile_proba(
                image_resolution,
                predictions,
                row,
                long_edge_ratio,
                short_edge_ratio,
                prediction_id_prefix=dataset_name + '/')
            # did not use extend here because for event
            # we want to use continuous frames instead of tiles
            track_to_fire_thresholds[track_id].append(
                min(tile_fire_thresholds))
    io_util.create_dir_if_not_exist(output_dir)
    output = {}
    output['track_to_fire_thresholds'] = track_to_fire_thresholds
    output['track_to_event_interval'] = track_to_event_interval
    output['predictions_thresholds'] = {
        k: v[1]
        for k, v in predictions.items()
    }
    tile_annotation_dir = datasets[dataset_name][1]
    ground_truth = io_util.load_all_pickles_from_dir(tile_annotation_dir)
    output['ground_truth'] = ground_truth
    with open(
            os.path.join(output_dir, '{}_event_recall.pkl'.format(
                action.replace('/', '_'))), 'wb') as f:
        pickle.dump(output, f)


def _get_grid_shape(dataset_name,
                    video_id,
                    long_edge_ratio=0.5,
                    short_edge_ratio=1):
    (im_w, im_h) = annotation_stats.dataset[dataset_name][
        'video_id_to_original_resolution'][video_id]
    if im_h > im_w:
        grid_h = int(1 / long_edge_ratio)
        grid_w = int(1 / short_edge_ratio)
    else:
        grid_h = int(1 / short_edge_ratio)
        grid_w = int(1 / long_edge_ratio)
    return (grid_w, grid_h)


def _get_tile_coords_iter(grid_w, grid_h):
    for grid_x in range(grid_w):
        for grid_y in range(grid_h):
            yield (grid_x, grid_y)


def get_prediction_id(dataset_name, video_id, frame_id, grid_x, grid_y):
    return '{}/{}_{}_{}_{}'.format(dataset_name, video_id, frame_id, grid_x,
                                   grid_y)


def get_predictions_by_ids(predictions, ids, ordered=True):
    if ordered:
        ids_to_predictions = collections.OrderedDict()
        for mid in ids:
            if mid in predictions:
                ids_to_predictions[mid] = predictions[mid]
    else:
        ids_to_predictions = {
            mid: predictions[mid]
            for mid in ids if mid in predictions
        }
    return ids_to_predictions


def get_max_positive_probability_from_prediction_dictionary(mdict):
    return max(mdict.iteritems(), key=lambda x: x[1][1])


def all_suppress_predictions():
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        logger.debug('working on {}'.format(dataset_name))
        suppressed_predictions = max_suppress_results(
            dataset_name,
            result_dir,
            span=30,
            stride=30,
            suppress_func=
            get_max_positive_probability_from_prediction_dictionary)
        result_base_dir = os.path.dirname(result_dir)
        dataset_output_dir = os.path.join(result_base_dir, 'suppress')
        io_util.create_dir_if_not_exist(dataset_output_dir)
        with open(os.path.join(dataset_output_dir, 'predictions.pkl'),
                  'wb') as f:
            pickle.dump(suppressed_predictions, f)


def get_test_dataset_video_iter(dataset_name):
    for video_id in annotation_stats.dataset[dataset_name]['test']:
        yield (dataset_name, video_id)
    for extra_test_dataset_name in annotation_stats.dataset[dataset_name][
            'extra_negative_dataset']:
        for video_id in annotation_stats.dataset[extra_test_dataset_name][
                'test']:
            yield (extra_test_dataset_name, video_id)


def max_suppress_results(
        dataset_name,
        result_dir,
        span=30,
        stride=30,
        suppress_func=get_max_positive_probability_from_prediction_dictionary):
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    suppressed_dict = {}
    test_data_iter = get_test_dataset_video_iter(dataset_name)
    for (test_dataset_name, video_id) in test_data_iter:
        video_id_to_frame_num = annotation_stats.dataset[test_dataset_name][
            'video_id_to_frame_num']
        video_id_to_original_resolution = annotation_stats.dataset[
            test_dataset_name]['video_id_to_original_resolution']
        frame_num = video_id_to_frame_num[video_id]
        (grid_w, grid_h) = _get_grid_shape(test_dataset_name, video_id)
        tile_coords_iter = _get_tile_coords_iter(grid_w, grid_h)
        for tile_coord in tile_coords_iter:
            logger.debug('working on {}/{} tile track: {}'.format(
                test_dataset_name, video_id, tile_coord))
            supress_start_frames = range(1, frame_num + 1, stride)
            for supress_start_frame in supress_start_frames:
                # find the max condition
                prediction_ids = [
                    get_prediction_id(test_dataset_name, video_id, frame_id,
                                      *tile_coord)
                    for frame_id in range(supress_start_frame,
                                          supress_start_frame + span)
                ]
                interval_ids_to_predictions = get_predictions_by_ids(
                    predictions, prediction_ids)
                for k, _ in interval_ids_to_predictions.iteritems():
                    suppressed_dict[k] = np.array([1.0, 0.0])
                survived_key, survived_val = suppress_func(
                    interval_ids_to_predictions)
                assert survived_val == max(
                    interval_ids_to_predictions.values(), key=lambda x: x[1])
                # logger.debug((survived_key, survived_val))
                suppressed_dict[survived_key] = survived_val
    return suppressed_dict


def has_any_fire(ids_to_predictions, threshold=0.5):
    prediction_positive_probas = [
        val[1] for val in ids_to_predictions.values()
    ]
    return (np.max(prediction_positive_probas) > threshold)


def has_all_fire(ids_to_predictions, threshold=0.5):
    prediction_positive_probas = [
        val[1] for val in ids_to_predictions.values()
    ]
    return (np.min(prediction_positive_probas) > threshold)


def set_all_predictions_to(ids_to_predictions, value):
    for mid in ids_to_predictions.keys():
        ids_to_predictions[mid] = value
    return ids_to_predictions


def all_interval_sampling_predictions(intervals=[
        1, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 92378 * 2
],
                                      thresholds=[0.1, 0.3]):
    for dataset_name, (annotation_dir, tile_annotation_dir,
                       result_dir) in datasets.iteritems():
        dataset_interval_to_predictions = {}
        for threshold in thresholds:
            interval_to_prediction_per_threshold = {}
            interval_fire_func = functools.partial(
                has_all_fire, threshold=threshold)
            for interval in intervals:
                logger.debug('working on {}, interval {}, threshold {}'.format(
                    dataset_name, interval, threshold))
                predictions_of_interval = interval_sampling_results(
                    dataset_name,
                    result_dir,
                    interval,
                    interval_fire_func=interval_fire_func)
                interval_to_prediction_per_threshold[
                    interval] = predictions_of_interval
            dataset_interval_to_predictions[
                threshold] = interval_to_prediction_per_threshold
        result_base_dir = os.path.dirname(result_dir)
        dataset_output_dir = os.path.join(result_base_dir, 'interval')
        io_util.create_dir_if_not_exist(dataset_output_dir)
        with open(os.path.join(dataset_output_dir, 'predictions.pkl'),
                  'wb') as f:
            pickle.dump(dataset_interval_to_predictions, f)


def get_all_test_tile_id_iter(dataset_name):
    test_data_iter = get_test_dataset_video_iter(dataset_name)
    for (test_dataset_name, video_id) in test_data_iter:
        video_id_to_frame_num = annotation_stats.dataset[test_dataset_name][
            'video_id_to_frame_num']
        frame_num = video_id_to_frame_num[video_id]
        (grid_w, grid_h) = _get_grid_shape(test_dataset_name, video_id)
        tile_coords_iter = _get_tile_coords_iter(grid_w, grid_h)

        for tile_coord in tile_coords_iter:
            for frame_id in range(1, frame_num + 1):
                tile_id = get_prediction_id(test_dataset_name, video_id,
                                            frame_id, *tile_coord)
                yield tile_id


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


def interval_sampling_results(dataset_name,
                              result_dir,
                              sample_interval,
                              interval_fire_func=has_any_fire):
    predictions = io_util.load_all_pickles_from_dir(result_dir)
    interval_sampling_prediction_dict = collections.OrderedDict()
    test_tile_id_iter = get_all_test_tile_id_iter(dataset_name)
    for tile_ids in grouper(sample_interval, test_tile_id_iter):
        interval_ids_to_predictions = get_predictions_by_ids(
            predictions, tile_ids, ordered=True)
        if interval_fire_func(interval_ids_to_predictions):
            # mark all tiles in the interval as 1
            set_all_predictions_to(interval_ids_to_predictions, 1.0)
        else:
            # mark all tiles in the interval as 0
            set_all_predictions_to(interval_ids_to_predictions, 0.0)
        interval_sampling_prediction_dict.update(interval_ids_to_predictions)
    return interval_sampling_prediction_dict


if __name__ == '__main__':
    fire.Fire()
