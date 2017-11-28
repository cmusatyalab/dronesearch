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

from itertools import izip_longest


def fix_data_format_in_redis():
    """Convert from string key to list of floats, not used"""
    r_server = redis.StrictRedis(
        host='localhost', port=6379, db=2)

    # iterate a list in batches of size n
    def batcher(iterable, n):
        args = [iter(iterable)] * n
        return izip_longest(*args)

    # in batches of 500 delete keys matching user:*
    batch_idx = 0
    batch_size = 10000
    for keybatch in batcher(r_server.scan_iter('*'), 10000):
        print('{}'.format((batch_idx + 1) * batch_size))
        outputs = r_server.mget(keybatch)
        outputs = [json.loads(result_str) for result_str in outputs]
        r_server.delete(*keybatch)
        for idx in range(len(keybatch)):
            r_server.lpush(keybatch[idx], outputs[idx])


def _get_class_prediction_for_exp1(r_server, ids):
    return r_server.mget(ids)


def _get_class_prediction_for_exp2(r_server, ids):
    """For 2_more_test experiments, we store both the softmax values
    and the 1024 mobilenet feature vectors. This method retrieve
    class predictions.

    Args:
      r_server: 
      ids: 

    Returns:

    """
    batch_size = 10000
    total_num = len(ids)
    batch_num = int(math.ceil(total_num * 1.0 / batch_size))
    predictions = []
    for batch_idx in range(batch_num):
        print('[{}/{}]'.format(((batch_idx + 1) * batch_size), total_num))
        batch_ids = ids[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        outputs = r_server.mget(batch_ids)
        outputs = [json.loads(result_str) for result_str in outputs]
        outputs = np.array(outputs)
        outputs = outputs[:, :2]
        predictions.extend(np.argmax(outputs, axis=1))
    return predictions


def print_stats(positive_dir,
                negative_dir,
                redis_db,
                get_prediction_fn=_get_class_prediction_for_exp2):
    r_server = redis.StrictRedis(
        host='localhost', port=6379, db=redis_db, socket_timeout=3600)
    class_label_map = {positive_dir: '1', negative_dir: '0'}
    tp, fn, tn, fp = 0, 0, 0, 0
    for class_dir in class_label_map.keys():
        print(class_dir)
        ids = [
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(class_dir)
        ]
        predictions = get_prediction_fn(r_server, ids)
        assert all([prediction is not None for prediction in predictions])
        cnt = collections.Counter(predictions)
        print([(cls, cnt[cls],
                '{:.2f}%'.format(cnt[cls] / len(predictions) * 100.0))
               for cls in cnt])
        if class_dir == positive_dir:
            tp = cnt['1']
            fn = cnt['0']
        else:
            tn = cnt['0']
            fp = cnt['1']
    recall = tp / (tp + fn * 1.0)
    precision = tp / (tp + fp * 1.0)
    fpr = fp / (fp + tn * 1.0)
    tpr = recall
    print('recall: {:.2f}, precision: {:.2f}, fpr: {:.2f}, tpr: {:.2f}'.format(
        recall, precision, fpr, tpr))


def write_wrong_predictions_to_file(input_dir, label, output_file_path,
                                    redis_db):
    r_server = redis.StrictRedis(host='localhost', port=6379, db=redis_db)
    ids = [
        os.path.splitext(file_name)[0] for file_name in os.listdir(input_dir)
    ]
    predictions = r_server.mget(ids)
    assert all([prediction is not None for prediction in predictions])
    # find wrong ids
    ids_wrong = []
    for idx, prediction in enumerate(predictions):
        if prediction != label:
            ids_wrong.append('{}.jpg'.format(ids[idx]))
    io_util.write_list_to_file(ids_wrong, output_file_path)


def write_false_positives_and_false_negatives(positive_dir, negative_dir,
                                              redis_db):
    class_label_map = {
        positive_dir: ['1', 'false_negative.txt'],
        negative_dir: ['0', 'false_positive.txt']
    }
    for class_dir, [label, output_file_path] in class_label_map.iteritems():
        print(class_dir)
        write_wrong_predictions_to_file(class_dir, label, output_file_path,
                                        redis_db)


def _get_prediction_results(imageid, slice_annotations, tile_coords, redis_db):
    r_server = redis.StrictRedis(host='localhost', port=6379, db=redis_db)
    prediction_keys = [
        slice_annotations.get_tile_file_basename_without_ext(
            imageid, tile_coord) for tile_coord in tile_coords
    ]
    return r_server.mget(prediction_keys)


def print_metrics_on_videos(image_dir, annotation_dir, video_list_file_path,
                            redis_db):
    # # image dir should be just the sliced test images
    # dataset = 'stanford'
    # slice_annotations = (
    #     annotation.SliceAnnotationsFactory.get_annotations_for_slices(
    #         dataset, image_dir, annotation_dir))

    # with open(video_list_file_path, 'r') as f:
    #     video_names = f.read().splitlines()
    # videoids = [
    #     video_name.replace('_video.mov', '') for video_name in video_names
    # ]

    # # filter by video name
    # annotations = slice_annotations.annotations
    # annotations = annotations[annotations['videoid'].isin(videoids)]

    # # filter by label
    # mask = annotation.get_positive_annotation_mask(annotations)
    # target_annotations = annotations[mask].copy()

    # track_annotations_grp = target_annotations.groupby(['trackid'])

    annotations = io_util.load_stanford_campus_annotation(annotation_dir)
    if video_list_file_path:
        annotations = annotation.filter_annotation_by_video_name(annotations,
                                                       video_list_file_path)
    annotations = annotation.filter_annotation_by_label(annotations)
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(annotations)

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
            bx = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            imageid = slice_annotations.get_imageid(row['videoid'],
                                                    row['frameid'])
            tile_coords = slice_annotations.get_tiles_contain_bounding_box(
                imageid, bx)
            predictions = _get_prediction_results(imageid, slice_annotations,
                                                  tile_coords, redis_db)
            # print('predictions for tiles {} are {}'.format(tile_coords,
            #                                                predictions))
            predictions_ints = [int(prediction) for prediction in predictions]
            # some positive is fired
            if sum(predictions_ints) != 0:
                print('first frame to be send: {}'.format(
                    start_frame, row['frameid']))
                detected_tracks[track_id] = row['frameid'] - start_frame
                break

    missed_tracks = set(all_tracks) - set(detected_tracks.keys())
    print('# missed tracks {}, percentage {:.2f}'.format(
        len(missed_tracks), len(missed_tracks) * 1.0 / len(all_tracks)))
    print('Average frame latency among detected tracks: {:.2f}({:.2f})'.format(
        np.mean(detected_tracks.values()), np.std(detected_tracks.values())))
    print(('Average time latency(s) among '
           'detected tracks (assuming 30FPS): {:.2f}({:.2f})').format(
               np.mean(detected_tracks.values()) / 30.0,
               np.std(detected_tracks.values()) / 30.0))


if __name__ == '__main__':
    fire.Fire()
