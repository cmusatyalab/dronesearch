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
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    threshold = 0.5
    stats = event_recall_stats[dataset_name]
    track_to_fire_thresholds = stats['track_to_fire_thresholds']
    track_to_event_fraction = {
        k: (np.array(v) > threshold).sum() / len(v)
        for k, v in track_to_fire_thresholds.items()
    }
    event_fractions = track_to_event_fraction.values()
    n, bins, patches = ax.hist(event_fractions, 10, alpha=0.75)
    ax.set_xlabel('Event Fraction')
    ax.set_ylabel('Number of Event')
    plt.savefig(
        'fig-event-fraction-{}.pdf'.format(
            dataset_name),
        bbox_inches='tight')

# with open('plotted_event_fraction.pkl', 'wb') as f:
#     pickle.dump(event_recall_to_disk, f)
