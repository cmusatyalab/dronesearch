"""Analyze stanford results."""
from __future__ import absolute_import, division, print_function

import collections
import os
import numpy as np

import annotation
import fire
import io_util
import redis


def print_stats(positive_dir, negative_dir):
    r_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    # class_dirs = [positive_dir, negative_dir]
    class_label_map = {positive_dir: '1', negative_dir: '0'}
    for class_dir in class_label_map.keys():
        print(class_dir)
        ids = [
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(class_dir)
        ]
        predictions = r_server.mget(ids)
        assert all([prediction is not None for prediction in predictions])
        cnt = collections.Counter(predictions)
        print([(cls, cnt[cls],
                '{:.2f}%'.format(cnt[cls] / len(predictions) * 100.0))
               for cls in cnt])


def write_wrong_predictions_to_file(input_dir, label, output_file_path):
    r_server = redis.StrictRedis(host='localhost', port=6379, db=0)
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


def write_false_positives_and_false_negatives(positive_dir, negative_dir):
    class_label_map = {
        positive_dir: ['1', 'false_negative.txt'],
        negative_dir: ['0', 'false_positive.txt']
    }
    for class_dir, [label, output_file_path] in class_label_map.iteritems():
        print(class_dir)
        write_wrong_predictions_to_file(class_dir, label, output_file_path)


def _get_prediction_results(imageid, slice_annotations, tile_coords):
    r_server = redis.StrictRedis(host='localhost', port=6379, db=0)
    prediction_keys = [
        slice_annotations.get_tile_file_basename_without_ext(
            imageid, tile_coord) for tile_coord in tile_coords
    ]
    return r_server.mget(prediction_keys)


def print_metrics_on_videos(image_dir, annotation_dir, video_list_file_path):
    # image dir should be just the sliced test images
    dataset = 'stanford'
    slice_annotations = (
        annotation.SliceAnnotationsFactory.get_annotations_for_slices(
            dataset, image_dir, annotation_dir))

    with open(video_list_file_path, 'r') as f:
        video_names = f.read().splitlines()
    videoids = [
        video_name.replace('_video.mov', '') for video_name in video_names
    ]

    # filter by video name
    annotations = slice_annotations.annotations
    annotations = annotations[annotations['videoid'].isin(videoids)]

    # filter by label
    mask = annotation.get_positive_annotation_mask(annotations)
    target_annotations = annotations[mask].copy()

    track_annotations_grp = target_annotations.groupby(['trackid'])
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
                                                  tile_coords)
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
