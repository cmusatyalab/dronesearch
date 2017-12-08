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
import matplotlib.pyplot as plt

random_sample_intervals = [1, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000]
for dataset_name in result_analysis.datasets.keys():
    print('working on {}'.format(dataset_name))
    interval_to_event_recall = result_analysis.random_select_accuracy(
        dataset_name, random_sample_intervals)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    frame_percent = 1.0 / np.array(interval_to_event_recall.keys())
    event_recall = interval_to_event_recall.values()
    ax.plot(event_recall, frame_percent, 'rs-')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlim(1.0, 0)
    ax.set_xlabel('Event Recall')
    ax.set_ylabel('Fraction of Frames Sent')
    ax.set_aspect('equal')
    ax.grid(linestyle='--', linewidth=1)
    plt.savefig(
        'fig-random-select-{}.pdf'.format(dataset_name), bbox_inches='tight')
