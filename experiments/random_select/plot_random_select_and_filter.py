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
experiment_name = 'random_select_and_filter'
cache_file_name = 'cache_{}.pkl'.format(experiment_name)


def _load_cache():
    if not os.path.exists(cache_file_name):
        raise ValueError('No cache file found.')
    else:
        with open(cache_file_name, 'rb') as f:
            dataset_stats = pickle.load(f)
    return dataset_stats


def _calc_dataset_stats():
    dataset_stats = {}
    for dataset_name in plot_util.experiments[experiment_name]:
        predictions = plot_util.get_predictions(experiment_name, dataset_name)
        interval_to_frames_event_recalls = collections.defaultdict(list)
        for interval, interval_predictions in predictions.iteritems():
            print('working on {}, interval {}'.format(dataset_name, interval))
            threshold_list = list(np.logspace(-3, -1, num=100, endpoint=True))
            threshold_list.extend(list(np.arange(0.1, 0.9, 0.01)))
            threshold_list.extend(
                list(1 - np.logspace(-6, -1, num=100, endpoint=True))[::-1])
            # threshold_list.extend(list(np.arange(0.91, 1.01, 0.01)))

            # print('threshold list: {}'.format(threshold_list))
            event_recalls_per_interval, frames_transmitted, threshold_list = result_analysis.get_event_recall(
                dataset_name, interval_predictions, threshold_list)
            #                list(np.arange(0, 1.01, 0.01)))
            interval_to_frames_event_recalls[interval] = (
                event_recalls_per_interval, frames_transmitted, threshold_list)
        dataset_stats[dataset_name] = interval_to_frames_event_recalls

    with open(cache_file_name, 'wb') as f:
        pickle.dump(dataset_stats, f)
    return dataset_stats


def _plot_line_graph_event_recall_to_bw(dataset_stats):
    datasets = collections.OrderedDict([
        ('okutama', 'T1'),
        ('stanford', 'T2'),
        ('raft', 'T3'),
        ('elephant', 'T4'),
    ])
    N = len(dataset_stats['elephant'])
    for dataset_idx, dataset_name in enumerate(datasets.keys()):
        cmap = plt.cm.rainbow(np.linspace(0, 1, N))
        color = iter(cmap)
        plt.clf()
        fig = plt.figure(figsize=(5, 10))
        ax = fig.add_subplot(1, 1, 1)

        print('plotting {}'.format(dataset_name))
        for interval, (event_recalls, frames_transmitted,
                       threshold_list) in sorted(
                           dataset_stats[dataset_name].iteritems(),
                           key=lambda x: x[0]):
            # if interval in [600]:
            #     continue

            # if dataset_name == 'elephant' and interval == 100:
            #     import pdb
            #     pdb.set_trace()

            event_recalls = event_recalls[:-1]
            frames_transmitted = frames_transmitted[:-1]
            ax.plot(
                event_recalls,
                frames_transmitted,
                color=next(color),
                label='Sample Interval {} '.format(interval))
        ax.legend(loc='center right', ncol=1, bbox_to_anchor=(2.5, 0.5))
        ax.set_xlabel('Event Recall')
        ax.set_ylabel('Frames')
        ax.set_yscale("log", nonposy='clip')
        ax.set_ylim(bottom=0.1)
        ax.tick_params(axis='y', which='both', bottom='off')
        plt.savefig(
            'fig-random-select-and-filter-recall-frame-{}.pdf'.format(
                dataset_name),
            bbox_inches='tight')


def _plot_aggregated_line_graph_event_recall_to_bw(dataset_stats):
    datasets = collections.OrderedDict([
        ('okutama', 'T1'),
        ('stanford', 'T2'),
        ('raft', 'T3'),
        ('elephant', 'T4'),
    ])
    N = 2
    for dataset_idx, dataset_name in enumerate(datasets.keys()):
        cmap = plt.cm.rainbow(np.linspace(0, 1, N))
        color = iter(cmap)
        plt.clf()
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)

        print('plotting {}'.format(dataset_name))
        aggregated = collections.defaultdict(list)
        baseline_aggregated = collections.defaultdict(list)
        for interval, (event_recalls, frames_transmitted,
                       threshold_list) in sorted(
                           dataset_stats[dataset_name].iteritems(),
                           key=lambda x: x[0]):
            for event_recall_idx, event_recall in enumerate(event_recalls):
                if frames_transmitted[event_recall_idx] < 10e-6:
                    continue
                aggregated[event_recall].append(
                    frames_transmitted[event_recall_idx])
            baseline_aggregated[event_recalls[0]].append(frames_transmitted[0])
        min_aggregated = {}
        global_min = 100000000
        for event_recall, frames_from_intervals in sorted(
                aggregated.iteritems(), key=lambda x: x[0], reverse=True):
            global_min = min(global_min, min(frames_from_intervals))
            min_aggregated[event_recall] = global_min

        sorted_aggregated = sorted(
            min_aggregated.iteritems(), key=lambda x: x[0])
        event_recalls, frames_transmitted = zip(*sorted_aggregated)

        global_min = 100000000
        baseline_min_aggregated = {}
        for event_recall, frames_from_intervals in sorted(
                baseline_aggregated.iteritems(), key=lambda x: x[0],
                reverse=True):
            global_min = min(global_min, min(frames_from_intervals))
            baseline_min_aggregated[event_recall] = global_min
        sorted_baseline_event_recalls, baseline_frames = zip(
            *sorted(baseline_min_aggregated.iteritems(), key=lambda x: x[0]))
        # event_recalls = event_recalls[:-1]
        # frames_transmitted = frames_transmitted[:-1]
        ax.plot(
            sorted_baseline_event_recalls,
            np.array(baseline_frames) /
            annotation_stats.dataset[dataset_name]['total_test_frames'],
            color='b',
            label='Sample')
        ax.plot(
            event_recalls,
            np.array(frames_transmitted) /
            annotation_stats.dataset[dataset_name]['total_test_frames'],
            color='r',
            label='Sample + Filter')

        # ax.legend(loc='center right', ncol=1, bbox_to_anchor=(2.0, 0.5))
        # ax.legend(loc='best')
        ax.set_xlabel('Event Recall')
        ax.set_ylabel('Frame Fraction')
        ax.set_yscale("log", nonposy='clip')
        # ax.set_ylim(bottom=0.1)
        ax.tick_params(axis='y', which='both', bottom='off')
        plt.savefig(
            'fig-random-select-and-filter-recall-frame-{}-aggregated.pdf'.
            format(dataset_name),
            bbox_inches='tight')


if __name__ == '__main__':
    if use_cache:
        dataset_stats = _load_cache()
    else:
        dataset_stats = _calc_dataset_stats()
    # _plot_line_graph_event_recall_to_bw(dataset_stats)
    _plot_aggregated_line_graph_event_recall_to_bw(dataset_stats)

    figlegend = plt.figure(figsize=(4, 1))
    import matplotlib.lines as mlines
    colors = ['b', 'r']
    labels = ['Sample', 'Sample + Filter']
    lines = []
    for color_idx, color in enumerate(colors):
        lines.append(
            mlines.Line2D(
                [0, 0], [1, 0],
                color=color,
                label=labels[color_idx],
                linestyle='-',
                linewidth=1))
    figlegend.legend(lines, labels, 'center', ncol=2)
    figlegend.savefig(
        'fig-recall-frame-aggregated-legend.pdf', bbox_inches='tight')
