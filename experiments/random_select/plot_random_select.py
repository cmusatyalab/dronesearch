from __future__ import absolute_import, division, print_function

import cPickle as pickle
import collections
import json
import os
import numpy as np
import sys
sys.path.insert(0, '../../scripts')
import plot_util
import result_analysis
import annotation_stats
import matplotlib.pyplot as plt

use_cache = bool(int(sys.argv[1]))
experiment_name = 'random_select'


def _load_cache():
    if not os.path.exists('cache.pkl'):
        raise ValueError('No cache file found.')
    else:
        with open('cache.pkl', 'rb') as f:
            dataset_stats = pickle.load(f)
    return dataset_stats


def _calc_dataset_stats():
    dataset_stats = {}
    for dataset_name in plot_util.experiments[experiment_name]:
        predictions = plot_util.get_predictions(experiment_name, dataset_name)
        interval_to_frames_event_recalls = collections.defaultdict(list)
        for interval, interval_predictions in predictions.iteritems():
            print('working on {}, interval {}'.format(dataset_name, interval))
            # since the generated probability for random select is 1
            # therefore we use 0.5 as filtering
            event_recalls_per_interval, _, _ = result_analysis.get_event_recall(
                dataset_name, interval_predictions, [0.5])
            interval_to_frames_event_recalls[interval] = (
                event_recalls_per_interval[0], len(interval_predictions))
        dataset_stats[dataset_name] = interval_to_frames_event_recalls

    with open('cache.pkl', 'wb') as f:
        pickle.dump(dataset_stats, f)
    return dataset_stats


def _plot_bar_graph_interval_to_event_recall(dataset_stats):
    datasets = collections.OrderedDict([
        ('okutama', 'T1'),
        ('stanford', 'T2'),
        ('raft', 'T3'),
        ('elephant', 'T4'),
    ])
    N = len(dataset_stats['elephant'])
    ind = np.arange(N)
    width = 0.15
    cmap = plt.cm.rainbow(np.linspace(0, 1, N))
    color = iter(cmap)
    plt.clf()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    rects = []
    for dataset_idx, dataset_name in enumerate(datasets.keys()):
        print('plotting {}'.format(dataset_name))
        event_recalls = []
        for interval, (event_recall, _) in sorted(
                dataset_stats[dataset_name].iteritems(), key=lambda x: x[0]):
            event_recalls.append(event_recall)
        rect = ax.bar(
            ind + width * dataset_idx, event_recalls, width, color=next(color))
        rects.append(rect)
    ax.legend(
        rects,
        datasets.values(),
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 1.25))
    ax.set_xlabel('Sample Interval')
    ax.set_ylabel('Event Recall')
    ax.set_xticks(ind + width * float(len(datasets)) / 2.0)
    ax.set_xticklabels(sorted(dataset_stats['elephant'].keys()))
    plt.savefig('fig-random-select-interval-recall.pdf', bbox_inches='tight')


def _plot_line_graph_event_recall_to_bw(dataset_stats):
    datasets = collections.OrderedDict([
        ('okutama', 'T1'),
        ('stanford', 'T2'),
        ('raft', 'T3'),
        ('elephant', 'T4'),
    ])
    N = len(datasets)
    cmap = plt.cm.rainbow(np.linspace(0, 1, N))
    color = iter(cmap)
    plt.clf()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    for dataset_idx, dataset_name in enumerate(datasets.keys()):
        print('plotting {}'.format(dataset_name))
        event_recall_to_frames = collections.OrderedDict()
        for interval, (event_recall, _) in sorted(
                dataset_stats[dataset_name].iteritems(), key=lambda x: x[0]):
            frame_num = int(
                annotation_stats.dataset[dataset_name]['total_test_frames'] /
                interval)
            if event_recall in event_recall_to_frames:
                if frame_num < event_recall_to_frames[event_recall]:
                    event_recall_to_frames[event_recall] = frame_num
            else:
                event_recall_to_frames[event_recall] = frame_num
        ax.plot(
            event_recall_to_frames.keys(),
            event_recall_to_frames.values(),
            color=next(color),
            label=datasets[dataset_name])
    ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.25))
    ax.set_xlabel('Event Recall')
    ax.set_ylabel('Frames')
    # ax.set_xticks(ind + width * float(len(datasets)) / 2.0)
    # ax.set_xticklabels(sorted(dataset_stats['elephant'].keys()))
    plt.savefig('fig-random-select-recall-frame.pdf', bbox_inches='tight')


if __name__ == '__main__':
    if use_cache:
        dataset_stats = _load_cache()
    else:
        dataset_stats = _calc_dataset_stats()
    # _plot_bar_graph_interval_to_event_recall(dataset_stats)
    _plot_line_graph_event_recall_to_bw(dataset_stats)

# event_recalls, transmitted_frames = zip(*sorted_frames_event_recall_tuple)

#     ax.plot(event_recalls, transmitted_frames, 'rs-')
#     # ax.set_xticks(np.arange(0, 1.1, 0.1))
#     # ax.set_yticks(np.arange(0, 1.1, 0.1))
#     # ax.set_xlim(1.0, 0)
#     ax.set_xlabel('Event Recall')
#     ax.set_ylabel('Frames Sent')
#     # ax.set_aspect('equal')
#     # ax.grid(linestyle='--', linewidth=1)
