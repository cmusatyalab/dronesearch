from __future__ import absolute_import, division, print_function

import cPickle as pickle
import collections
import numpy as np
import sklearn.metrics

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "serif",
    "font.serif": [],  # use latex default serif font
    "font.sans-serif": ["DejaVu Sans"],  # use a specific sans-serif font
    "font.size": 25
}
mpl.rcParams.update(pgf_with_rc_fonts)
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fix_prediction_id_to_ground_truth_id(prediction_id):
    """

    Args:
      prediction_id: 

    Returns:

    """
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
cmap = plt.cm.rainbow(np.linspace(0, 1, len(dataset_to_legend.keys())))
color = iter(cmap)

fig, ax1 = plt.subplots()
for dataset_name in ['okutama', 'stanford', 'raft', 'elephant']:
    stats = event_recall_stats[dataset_name]
    print('ploting {}'.format(dataset_name))
    track_to_fire_thresholds = stats['track_to_fire_thresholds']
    track_to_min_fire_thresholds = {
        k: max(v)
        for k, v in track_to_fire_thresholds.items()
    }
    fire_thresholds = track_to_min_fire_thresholds.values()
    fire_thresholds = np.append(fire_thresholds, 0)
    fire_thresholds = sorted(fire_thresholds, reverse=True)

    event_recall = np.array(range(
        1,
        len(fire_thresholds) + 1)) / len(fire_thresholds)
    c_color = next(color)
    ax1.plot(fire_thresholds, event_recall, c=c_color, label='Event Recall')
    break

ax1.set_xlabel('Cutoff Threshold')
ax1.set_ylabel('Event Recall', color='b')
ax1.tick_params('y', colors='b')
ax1.tick_params('y')
ax1.set_xlim([0, 1.05])
ax1.set_ylim([0, 1.05])

ax2 = ax1.twinx()

print('finished plotting event recall vs threshold')

for dataset_name in ['okutama', 'stanford', 'raft', 'elephant']:
    tats = event_recall_stats[dataset_name]
    color = iter(cmap)
    print('ploting {}'.format(dataset_name))
    predictions_thresholds = stats['predictions_thresholds']
    increasing_threshold_predictions = collections.OrderedDict(
        sorted(predictions_thresholds.items(), key=lambda t: t[1]))
    assert len(increasing_threshold_predictions) == len(predictions_thresholds)
    ground_truths = stats['ground_truth']
    y_true, y_score = [], []
    positive_num, negative_num = 0, 0
    for tile_id, threshold in increasing_threshold_predictions.iteritems():
        if ground_truths[_fix_prediction_id_to_ground_truth_id(tile_id)]:
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

    ax2.plot(roc_threshold, tp + fp, 'r-', label='Transmitted Frames %')
    ax2.set_ylabel('Percentage of Frames', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    break

# plt.legend(loc='best', fontsize=22)
plt.gca().invert_xaxis()

axins = zoomed_inset_axes(
    ax1,
    1,
    loc='upper left',
    bbox_to_anchor = (100,800)
)
axins.plot(fire_thresholds, event_recall, c=c_color, label='Event Recall')
axins.plot(fire_thresholds, event_recall, c=c_color, marker='o')
axins.plot(roc_threshold, tp + fp, 'r-')
x1, x2, y1, y2 = 0.9, 1.02, 0, 1
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#mark_inset(ax1, axins, loc1=1, loc2=3)
plt.gca().invert_xaxis()
plt.savefig(
    'fig-event-recall-frame-percentage-vs-threshold.pdf', bbox_inches='tight')
