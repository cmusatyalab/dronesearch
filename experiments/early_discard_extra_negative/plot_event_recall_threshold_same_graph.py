from __future__ import absolute_import, division, print_function

import cPickle as pickle
import collections
import json
import os
import numpy as np

import matplotlib as mpl
import sklearn.metrics
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset,
                                                   zoomed_inset_axes)

mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "font.size": 25
}
mpl.rcParams.update(pgf_with_rc_fonts)
# mpl.use('Agg')
import matplotlib.pyplot as plt


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


with open('event_recall.pkl') as f:
    event_recall_stats = pickle.load(f)

dataset_to_legend = {
    'elephant': 'T5',
    'raft': 'T4',
    'okutama': 'T1',
    'stanford': 'T3'
}
cmap = plt.cm.rainbow(np.linspace(0, 1, 3))

event_recall_to_disk = {}
for dataset_name in ['okutama', 'stanford', 'raft', 'elephant']:
    plt.clf()
    fig, ax2 = plt.subplots()
    print('plotting frame percentage vs threshold for {}'.format(dataset_name))
    stats = event_recall_stats[dataset_name]
    predictions_thresholds = stats['predictions_thresholds']
    increasing_threshold_predictions = collections.OrderedDict(
        sorted(predictions_thresholds.items(), key=lambda t: t[1]))
    assert len(increasing_threshold_predictions) == len(predictions_thresholds)
    ground_truths = collections.defaultdict(bool, stats['ground_truth'])
    y_true, y_score = [], []
    positive_num, negative_num = 0, 0
    for tile_id, threshold in increasing_threshold_predictions.iteritems():
        gt_id = _fix_prediction_id_to_ground_truth_id(tile_id)
        if ground_truths[gt_id]:
            positive_num += 1
        else:
            negative_num += 1
        y_true.append(
            ground_truths[_fix_prediction_id_to_ground_truth_id(tile_id)])
        y_score.append(threshold)
    print('positive num: {}, negative num: {}'.format(positive_num,
                                                      negative_num))
    fpr, tpr, roc_threshold = sklearn.metrics.roc_curve(y_true, y_score)
    fp = fpr * negative_num
    tp = tpr * positive_num
    fn = (1 - tpr) * positive_num
    tn = (1 - fpr) * negative_num

    total_num = positive_num + negative_num
    fp = fp / total_num
    tp = tp / total_num
    fn = fn / total_num
    tn = tn / total_num

    roc_threshold = roc_threshold[::-1]
    tp = tp[::-1]
    fp = fp[::-1]
    ax2.plot(roc_threshold, tp + fp, 'r-', label='Transmitted')
    # ax2.plot(roc_threshold, tp + fp, 'ro')
    color = iter(cmap)
    ax2.fill_between(
        roc_threshold,
        tp,
        0,
        where=tp >= 0,
        facecolor=next(color),
        alpha=0.5,
        interpolate=True,
        label='True Positives')
    ax2.fill_between(
        roc_threshold,
        fp + tp,
        tp,
        where=fp + tp > tp,
        facecolor=next(color),
        alpha=0.5,
        interpolate=True,
        label='False Positives')
    ax2.set_ylabel('Frame %', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        fancybox=True,
        fontsize=20)

    print('\n\nploting event recall vs threshold for {}'.format(dataset_name))
    ax1 = ax2.twinx()
    track_to_fire_thresholds = stats['track_to_fire_thresholds']
    track_to_min_fire_thresholds = {
        k: max(v) if v else 0.0
        for k, v in track_to_fire_thresholds.items()
    }
    print(json.dumps(track_to_min_fire_thresholds, indent=4))
    fire_thresholds = track_to_min_fire_thresholds.values()
    fire_thresholds = sorted(fire_thresholds, reverse=True)
    event_recall = np.array(range(
        1,
        len(fire_thresholds) + 1)) / len(fire_thresholds)
    fire_thresholds = fire_thresholds[::-1]
    event_recall = event_recall[::-1]
    ax1.plot(fire_thresholds, event_recall, 'bs')
    event_recall_to_disk[dataset_name] = (fire_thresholds, event_recall)

    fire_thresholds = np.insert(fire_thresholds, 0, 0)
    event_recall = np.insert(event_recall, 0, 1)
    ax1.plot(fire_thresholds, event_recall, 'b-', label='Event Recall')
    ax1.set_xlabel('Cutoff Threshold')
    ax1.set_ylabel('Event Recall', color='b')
    ax1.tick_params('y', colors='b')
    ax1.tick_params('y')
    ax1.set_xlim([0, 1.05])
    ax1.set_ylim([0, 1.05])

    print('finished plotting event recall vs threshold. {} events'.format(
        len(fire_thresholds)))

    plt.savefig(
        'fig-event-recall-frame-percentage-vs-threshold-{}.pdf'.format(
            dataset_name),
        bbox_inches='tight')

with open('plotted_event_recall.pkl', 'wb') as f:
    pickle.dump(event_recall_to_disk, f)
