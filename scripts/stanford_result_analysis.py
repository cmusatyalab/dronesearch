"""Analyze stanford results."""
from __future__ import absolute_import, division, print_function

import collections
import os

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
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(input_dir)
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
    class_label_map = {positive_dir: ['1', 'false_negative.txt'], negative_dir:
                       ['0', 'false_positive.txt']}
    for class_dir, [label, output_file_path] in class_label_map.iteritems():
        print(class_dir)
        write_wrong_predictions_to_file(class_dir, label, output_file_path)


if __name__ == '__main__':
    fire.Fire()
