import itertools

import fire
import os

import pandas as pd
import numpy as np
from sklearn.utils import resample

import annotation_stats

import matplotlib

from jitl_data import _split_imageid
from jitl_test import get_video_frame_to_uniq_track_id

matplotlib.use('Agg')
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 16})


def frames_vs_dnn_cutoff(jitl_result_file,
                         savefig=None):
    cache_file = None
    if savefig:
        cache_file = savefig + '-frames_vs_dnn_cutoff.cache'

    if savefig \
            and os.path.exists(cache_file) \
            and os.path.getmtime(cache_file) > os.path.getmtime(jitl_result_file):
        print("Plot loading cache {}".format(cache_file))
        df = pd.read_pickle(cache_file)

    else:
        df = _calc_dnn_cutoff_frames_tp_dataframe(jitl_result_file)

        if cache_file:
            print("Plot writing cache {}".format(cache_file))
            df.to_pickle(cache_file)

    df['dnn_precision'] = (np.array(df['n_dnn_tp']) / np.array(df['n_dnn_fire'])).tolist()
    df['jitl_precision'] = ((np.array(df['n_jitl_tp']) / np.array(df['n_jitl_fire'])) + 0.01).tolist()

    print("Will be plotting from ...")
    print(df)

    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_xlabel("DNN cutoff probability")
    ax1.set_ylabel("# transmitted frames")
    ax2 = ax1.twinx()

    # Plot precision TP/(TP+FP)

    ax2.plot(df['dnn_cutoff'], df['dnn_precision'], 'b^:', label="DNN precision")
    ax2.plot(df['dnn_cutoff'], df['jitl_precision'] + 0.05, 'r^:', label="JITL precision")
    ax2.set_ylim(bottom=0, top=1.05)

    # Plot # frames
    ax1.plot(df['dnn_cutoff'], df['n_dnn_fire'], 'bo-', label="DNN all")
    ax1.plot(df['dnn_cutoff'], df['n_dnn_tp'], 'bo--', label="DNN true positive")
    ax1.plot(df['dnn_cutoff'], df['n_jitl_fire'] - 1, 'ro-', label="JITL all")
    ax1.plot(df['dnn_cutoff'], df['n_jitl_tp'] - 1, 'ro--', label="JITL true positive")
    ax1.set_ylim(bottom=0)

    plt.legend(loc='upper left')
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    plt.show()


def _calc_dnn_cutoff_frames_tp_dataframe(jitl_result_file):
    result_df = pd.read_pickle(jitl_result_file)
    dnn_cutoff_grps = result_df.groupby(['dnn_cutoff'])
    df = pd.DataFrame()
    for dnn_cutoff, results in dnn_cutoff_grps:
        print(dnn_cutoff)
        print(results)

        n_dnn_fire = results['label'].map(lambda x: len(x)).sum()
        n_dnn_tp = results['label'].map(lambda x: np.count_nonzero(x)).sum()
        assert n_dnn_tp <= n_dnn_fire
        n_jitl_fire = results['jitl_prediction'].map(lambda x: np.count_nonzero(x)).sum()
        assert n_jitl_fire <= n_dnn_fire
        n_jitl_tp = results.apply(
            lambda row: np.count_nonzero(np.logical_and(row['jitl_prediction'], row['label'])), axis=1).sum()
        assert n_jitl_tp <= n_jitl_fire

        df = df.append({
            'dnn_cutoff': dnn_cutoff,
            'n_dnn_fire': n_dnn_fire,
            'n_dnn_tp': n_dnn_tp,
            'n_jitl_fire': n_jitl_fire,
            'n_jitl_tp': n_jitl_tp
        }, ignore_index=True)
    return df


def event_recall_vs_dnn_cutoff(base_dir,
                               dataset,
                               jitl_result_file,
                               savefig=None):
    assert dataset in annotation_stats.dataset.keys()

    df = _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file)

    print("Will be plotting from ...")
    print(df)

    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_xlabel("DNN cutoff probability")
    ax1.set_ylabel("Event recall")

    df = df.sort_values(by=['dnn_cutoff'])

    ax1.plot(df['dnn_cutoff'], df['dnn_event_recall'], 'b-', label='DNN')
    ax1.plot(df['dnn_cutoff'], df['jitl_event_recall'], 'r-', label='JITL')

    ax1.set_ylim(bottom=0, top=1.1)

    plt.legend()
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    plt.show()


def frames_vs_event_recall(base_dir,
                           dataset,
                           jitl_result_file,
                           random_drop=True,
                           savefig=None):
    assert dataset in annotation_stats.dataset.keys()

    cache_file = None
    if savefig:
        cache_file = savefig + '-frames_vs_event_recall-{}.cache'.format(random_drop)

    if cache_file and os.path.exists(cache_file) \
            and os.path.getmtime(cache_file) > os.path.getmtime(jitl_result_file):
        print("Plot loading cache {}".format(cache_file))
        df = pd.read_pickle(cache_file)

    else:
        df = _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file, random_drop)
        if cache_file:
            print("Plot writing cache {}".format(cache_file))
            df.to_pickle(cache_file)

    print("Will be plotting from ...")
    print(df)

    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_xlabel("Event recall")
    ax1.set_ylabel("# transmitted frames")

    df1 = df[['dnn_event_recall', 'dnn_fired_frames']]
    df1 = df1.groupby(['dnn_event_recall']).aggregate(min)  # crunch duplicated recall values
    df1['dnn_event_recall'] = df1.index
    df1 = df1.sort_values(by=['dnn_event_recall'])
    # df1 = df1.sort_index()
    ax1.plot(df1['dnn_event_recall'], df1['dnn_fired_frames'], 'bo-', label='DNN')

    df1 = df[['jitl_event_recall', 'jitl_fired_frames']]
    df1 = df1.groupby(['jitl_event_recall']).aggregate(min)
    df1['jitl_event_recall'] = df1.index
    df1 = df1.sort_values(by=['jitl_event_recall'])
    ax1.plot(df1['jitl_event_recall'], df1['jitl_fired_frames'] - 1, 'ro-', label='JITL')

    if random_drop:
        df1 = df[['random_drop_event_recall', 'random_drop_fired_frames']]
        df1 = df1.groupby(['random_drop_event_recall']).aggregate(min)  # crunch duplicated recall values
        print("Random drop result:")
        print(df1)
        df1['random_drop_event_recall'] = df1.index
        df1 = df1.sort_values(by=['random_drop_event_recall'])
        ax1.plot(df1['random_drop_event_recall'], df1['random_drop_fired_frames'] + 1, 'go-', label='Random Drop')

    ax1.set_ylim(bottom=0)

    plt.legend(loc='lower right')
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    plt.show()


def _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file, random_drop=False, random_drop_repeat=5):
    all_unique_track_ids, video_frame_to_uniq_track_id = get_video_frame_to_uniq_track_id(base_dir, dataset)
    jitl_results = pd.read_pickle(jitl_result_file)
    # Starting analyzing event recall
    dnn_cutoff_grps = jitl_results.groupby(['dnn_cutoff'])
    df = pd.DataFrame()
    for dnn_cutoff, results in dnn_cutoff_grps:
        # # there's a mysterious bug only happen on some data ...
        # ValueError: could not broadcast input array from shape (135) into shape (7)
        # results['jitl_fired_imageids'] = results.apply(
        #     lambda r: list([r['imageids'][ind] for ind in np.nonzero(r['jitl_prediction'])[0]]),
        #     axis=1)
        jitl_fired_imageids = []
        for imageids, prediction in zip(results['imageids'], results['jitl_prediction']):
            jitl_fired_imageids.extend([imageids[ind] for ind in np.nonzero(prediction)[0]])
        dnn_fired_imageids = list(itertools.chain.from_iterable(results['imageids']))
        # jitl_fired_imageids = list(itertools.chain.from_iterable(results['jitl_fired_imageids']))
        assert len(jitl_fired_imageids) <= len(dnn_fired_imageids)

        dnn_fired_track_ids = _calc_fired_events(video_frame_to_uniq_track_id, dnn_fired_imageids)
        jitl_fired_track_ids = _calc_fired_events(video_frame_to_uniq_track_id, jitl_fired_imageids)
        assert len(jitl_fired_track_ids) <= len(dnn_fired_track_ids)

        dnn_event_recall = float(len(dnn_fired_track_ids)) / len(all_unique_track_ids)
        jitl_event_recall = float(len(jitl_fired_track_ids)) / len(all_unique_track_ids)
        assert jitl_event_recall <= dnn_event_recall

        dct = {'dnn_cutoff': dnn_cutoff,
               'dnn_event_recall': dnn_event_recall,
               'jitl_event_recall': jitl_event_recall,
               'dnn_fired_frames': len(dnn_fired_imageids),
               'jitl_fired_frames': len(jitl_fired_imageids)}

        if random_drop:
            # simulate random drop
            rd_recalls = []
            for i in range(random_drop_repeat):
                random_drop_imageids = resample(dnn_fired_imageids,
                                                n_samples=len(jitl_fired_imageids),
                                                replace=False,
                                                random_state=42 + i)
                assert len(random_drop_imageids) == len(jitl_fired_imageids)
                random_drop_fired_track_ids = _calc_fired_events(video_frame_to_uniq_track_id, random_drop_imageids)
                random_drop_recall = float(len(random_drop_fired_track_ids)) / len(all_unique_track_ids)
                rd_recalls.append(random_drop_recall)
            print("DNN cutoff: {}. Random drop recalls: {}".format(dnn_cutoff, rd_recalls))
            dct.update({'random_drop_event_recall': np.mean(rd_recalls),
                        'random_drop_fired_frames': len(jitl_fired_imageids)})

        df = df.append(dct, ignore_index=True)
    return df


# FIXME: this is wrong. we need event association with tiles, not just frames.
def _calc_fired_events(video_frame_to_uniq_track_id, imageids):
    fired_track_ids = set()
    for imageid in imageids:
        video_id, frame_id, _, _ = _split_imageid(imageid)
        track_ids = video_frame_to_uniq_track_id.get((video_id, frame_id))
        if track_ids:
            fired_track_ids.update(track_ids)
    return fired_track_ids


if __name__ == '__main__':
    fire.Fire()
