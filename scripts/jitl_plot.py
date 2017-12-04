import itertools

import fire
import os

import pandas as pd
import numpy as np
import annotation_stats
import annotation

import matplotlib
import collections

from jitl_data import _split_imageid
from jitl_data import datasets

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def frames_vs_dnn_cutoff(jitl_data_file,
                         jitl_result_file,
                         savefig=None):
    cache_file = None
    if savefig:
        cache_file = savefig + '-frames_vs_dnn_cutoff.cache'

    if savefig and os.path.exists(cache_file) \
        and os.path.getmtime(cache_file) > os.path.getmtime(jitl_result_file) \
            and os.path.getmtime(cache_file) > os.path.getmtime(jitl_data_file):
        print("Plot loading cache {}".format(cache_file))
        df = pd.read_pickle(cache_file)

    else:
        data_df = pd.read_pickle(jitl_data_file)
        result_df = pd.read_pickle(jitl_result_file)

        dnn_cutoff_grps = result_df.groupby(['dnn_cutoff'])

        df = pd.DataFrame()
        for dnn_cutoff, results in dnn_cutoff_grps:
            print(dnn_cutoff)
            print(results)
            results['label'] = results['imageids'].map(
                lambda x: [data_df[data_df['imageid'] == imgid]['label'].iat[0] for imgid in x])

            n_dnn_fire = results['label'].map(lambda x: len(x)).sum()
            n_dnn_tp = results['label'].map(lambda x: np.count_nonzero(x)).sum()
            assert n_dnn_tp < n_dnn_fire
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

            if cache_file:
                print("Plot writing cache {}".format(cache_file))
                df.to_pickle(cache_file)

    print("Will be plotting from ...")
    print(df)

    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_xlabel("DNN cutoff probability")
    ax1.set_ylabel("# transmitted frames")

    print("Plotting")
    ax1.plot(df['dnn_cutoff'], df['n_dnn_fire'], 'b-', label="DNN all")
    ax1.plot(df['dnn_cutoff'], df['n_dnn_tp'], 'b--', label="DNN true positive")
    ax1.plot(df['dnn_cutoff'], df['n_jitl_fire'], 'r-', label="JITL all")
    ax1.plot(df['dnn_cutoff'], df['n_jitl_tp'], 'r--', label="JITL true positive")
    ax1.set_ylim(bottom=0)

    plt.legend(loc='upper left')
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    plt.show()


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
                           savefig=None):
    assert dataset in annotation_stats.dataset.keys()

    cache_file = None
    if savefig:
        cache_file = savefig + '-frames_vs_event_recall.cache'

    if cache_file and os.path.exists(cache_file) \
            and os.path.getmtime(cache_file) > os.path.getmtime(jitl_result_file):
        print("Plot loading cache {}".format(cache_file))
        df = pd.read_pickle(cache_file)

    else:
        df = _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file)
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
    ax1.plot(df1['jitl_event_recall'], df1['jitl_fired_frames'], 'ro-', label='JITL')

    ax1.set_ylim(bottom=0)

    plt.legend(loc='lower right')
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    plt.show()


def _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file):
    load_annotation_func = annotation_stats.dataset[dataset][
        'annotation_func']
    labels = annotation_stats.dataset[dataset]['labels']
    annotation_dir = os.path.join(base_dir, datasets[dataset][0])
    annotations = load_annotation_func(annotation_dir)
    test_video_ids = annotation_stats.dataset[dataset]['test']
    annotations = annotations[annotations['videoid'].isin(test_video_ids)]
    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    # make track ID unique across different videos
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)
    video_frame_to_uniq_track_id = collections.defaultdict(list)
    for track_id, track_annotations in track_annotations_grp:
        for _, row in track_annotations.iterrows():
            video_id = row['videoid']
            frame_id = row['frameid']
            video_frame_to_uniq_track_id[(video_id, frame_id)].append(track_id)
    print("Parsed annotations. Found {} unique track IDs in {}.".format(
        len(track_annotations_grp), ','.join(test_video_ids)))
    jitl_results = pd.read_pickle(jitl_result_file)
    # Starting analyzing event recall
    dnn_cutoff_grps = jitl_results.groupby(['dnn_cutoff'])
    df = pd.DataFrame()
    all_unique_trakc_ids = set(track_annotations_grp.groups.keys())
    for dnn_cutoff, results in dnn_cutoff_grps:
        results['jitl_fired_imageids'] = results.apply(
            lambda row: [row['imageids'][ind] for ind in np.nonzero(row['jitl_prediction'])[0]],
            axis=1)
        dnn_fired_imageids = list(itertools.chain.from_iterable(results['imageids']))
        jitl_fired_imageids = list(itertools.chain.from_iterable(results['jitl_fired_imageids']))

        dnn_fired_track_ids = _calc_fired_events(video_frame_to_uniq_track_id, dnn_fired_imageids)
        jitl_fired_track_ids = _calc_fired_events(video_frame_to_uniq_track_id, jitl_fired_imageids)
        assert len(jitl_fired_track_ids) <= len(dnn_fired_track_ids)

        dnn_event_recall = float(len(dnn_fired_track_ids)) / len(all_unique_trakc_ids)
        jitl_event_recall = float(len(jitl_fired_track_ids)) / len(all_unique_trakc_ids)
        assert jitl_event_recall <= dnn_event_recall

        df = df.append({
            'dnn_cutoff': dnn_cutoff,
            'dnn_event_recall': dnn_event_recall,
            'jitl_event_recall': jitl_event_recall,
            'dnn_fired_frames': len(dnn_fired_imageids),
            'jitl_fired_frames': len(jitl_fired_imageids)
        }, ignore_index=True)
    return df


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
