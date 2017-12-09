import collections
import itertools

import fire
import matplotlib
import numpy as np
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample

import annotation
import annotation_stats
from jitl_data import _split_imageid
from jitl_data import datasets, _combine_imageid
from result_analysis import _clamp_bbox, _get_tile_coords_from_bbox

matplotlib.use('Agg')
from matplotlib import pyplot as plt

# matplotlib.rcParams.update({'font.size': 16})

LABEL_FONTS = dict(fontsize=22)
LEGEND_FONTS = LABEL_FONTS
TICKS_FONTS = dict(fontsize=20)


def frames_vs_event_recall(base_dir,
                           dataset,
                           jitl_result_file,
                           random_drop=False,
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
    # plt.gca().invert_xaxis()
    ax1.set_xlabel("Event Recall", **LABEL_FONTS)
    ax1.set_ylabel("Frame Fraction", **LABEL_FONTS)

    df_dnn = df[['dnn_event_recall', 'dnn_fired_frames', 'total_test_frames']]
    df_dnn = df_dnn.groupby(['dnn_event_recall']).aggregate(min)  # crunch duplicated recall values
    df_dnn['dnn_event_recall'] = df_dnn.index
    df_dnn = df_dnn.sort_values(by=['dnn_event_recall'])
    df_dnn['dnn_fired_frames_percent'] = df_dnn['dnn_fired_frames'].astype(float) / df_dnn['total_test_frames']

    df_jitl = df[['jitl_event_recall', 'jitl_fired_frames', 'total_test_frames']]
    df_jitl = df_jitl.groupby(['jitl_event_recall']).aggregate(min)
    df_jitl['jitl_event_recall'] = df_jitl.index
    df_jitl = df_jitl.sort_values(by=['jitl_event_recall'])
    df_jitl['jitl_fired_frames_percent'] = df_jitl['jitl_fired_frames'].astype(float) / df_dnn['total_test_frames']

    print(df_dnn)
    print(df_jitl)

    shared_recalls = set(df_dnn['dnn_event_recall'].tolist()).intersection(df_jitl['jitl_event_recall'].tolist())
    df_dnn = df_dnn[df_dnn['dnn_event_recall'].isin(shared_recalls)]
    df_jitl = df_jitl[df_jitl['jitl_event_recall'].isin(shared_recalls)]

    ax1.plot(df_dnn['dnn_event_recall'], df_dnn['dnn_fired_frames_percent'], 'b-', label='DNN')
    ax1.plot(df_jitl['jitl_event_recall'], df_jitl['jitl_fired_frames_percent']-0.0001, 'r-', label='JITL')  # noisify overlap

    if random_drop:
        df1 = df[['random_drop_event_recall', 'random_drop_fired_frames']]
        df1 = df1.groupby(['random_drop_event_recall']).aggregate(min)  # crunch duplicated recall values
        print("Random drop result:")
        print(df1)
        df1['random_drop_event_recall'] = df1.index
        df1 = df1.sort_values(by=['random_drop_event_recall'])
        ax1.plot(df1['random_drop_event_recall'], df1['random_drop_fired_frames'] + 1, 'go-', label='Random Drop')

    ax1.set_ylim(bottom=0)
    # ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.legend(loc='lower right', **LEGEND_FONTS)
    plt.xticks(**TICKS_FONTS)
    plt.yticks(**TICKS_FONTS)
    ax1.get_xaxis().set_major_locator(MaxNLocator(4))

    plt.tight_layout()
    if savefig:
        print("Saving figure to {}".format(savefig))
        plt.savefig(savefig)

    # plt.show()


def _calc_cutoff_recall_frame_dataframe(base_dir, dataset, jitl_result_file, random_drop=False, random_drop_repeat=5):
    all_unique_track_ids, video_tile_to_uniq_track_id = get_video_tile_to_uniq_track_id(base_dir, dataset)
    jitl_results = pd.read_pickle(jitl_result_file)
    # Starting analyzing event recall
    dnn_cutoff_grps = jitl_results.groupby(['dnn_cutoff'])
    df = pd.DataFrame()
    for dnn_cutoff, results in dnn_cutoff_grps:
        jitl_fired_imageids = []
        for imageids, prediction in zip(results['imageids'], results['jitl_prediction']):
            jitl_fired_imageids.extend([imageids[ind] for ind in np.nonzero(prediction)[0]])
        dnn_fired_imageids = list(itertools.chain.from_iterable(results['imageids']))
        # jitl_fired_imageids = list(itertools.chain.from_iterable(results['jitl_fired_imageids']))
        assert len(jitl_fired_imageids) <= len(dnn_fired_imageids)

        dnn_fired_track_ids = _calc_fired_events(video_tile_to_uniq_track_id, dnn_fired_imageids)
        jitl_fired_track_ids = _calc_fired_events(video_tile_to_uniq_track_id, jitl_fired_imageids)
        assert len(jitl_fired_track_ids) <= len(dnn_fired_track_ids) <= len(all_unique_track_ids)

        print("all track ids: {}".format(','.join(all_unique_track_ids)))
        print("DNN fired track ids: {}".format(','.join(dnn_fired_track_ids)))
        print("JITL fired track ids: {}".format(','.join(jitl_fired_track_ids)))

        dnn_event_recall = float(len(dnn_fired_track_ids)) / len(all_unique_track_ids)
        jitl_event_recall = float(len(jitl_fired_track_ids)) / len(all_unique_track_ids)
        assert jitl_event_recall <= dnn_event_recall

        total_test_frames = annotation_stats.dataset[dataset]['total_test_frames'] * 2  # XXX we should actually tiles

        dct = {'dnn_cutoff': dnn_cutoff,
               'dnn_event_recall': dnn_event_recall,
               'jitl_event_recall': jitl_event_recall,
               'dnn_fired_frames': len(dnn_fired_imageids),
               'jitl_fired_frames': len(jitl_fired_imageids),
               'total_test_frames': total_test_frames}

        if random_drop:
            # simulate random drop
            rd_recalls = []
            for i in range(random_drop_repeat):
                random_drop_imageids = resample(dnn_fired_imageids,
                                                n_samples=len(jitl_fired_imageids),
                                                replace=False,
                                                random_state=42 + i)
                assert len(random_drop_imageids) == len(jitl_fired_imageids)
                random_drop_fired_track_ids = _calc_fired_events(video_tile_to_uniq_track_id, random_drop_imageids)
                random_drop_recall = float(len(random_drop_fired_track_ids)) / len(all_unique_track_ids)
                rd_recalls.append(random_drop_recall)
            print("DNN cutoff: {}. Random drop recalls: {}".format(dnn_cutoff, rd_recalls))
            dct.update({'random_drop_event_recall': np.mean(rd_recalls),
                        'random_drop_fired_frames': len(jitl_fired_imageids)})

        df = df.append(dct, ignore_index=True)
    return df


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


def _calc_fired_events(video_tile_to_uniq_track_id, imageids):
    fired_track_ids = set()
    for imageid in imageids:
        track_ids = video_tile_to_uniq_track_id.get(imageid)
        if track_ids:
            fired_track_ids.update(track_ids)
    return fired_track_ids


def get_video_tile_to_uniq_track_id(base_dir, dataset):
    load_annotation_func = annotation_stats.dataset[dataset][
        'annotation_func']
    labels = annotation_stats.dataset[dataset]['labels']
    annotation_dir = os.path.join(base_dir, datasets[dataset][0])
    annotations = load_annotation_func(annotation_dir)
    test_video_ids = annotation_stats.dataset[dataset]['test']
    annotations = annotations[annotations['videoid'].isin(test_video_ids)]
    annotations = annotation.filter_annotation_by_label(
        annotations, labels=labels)
    video_id_to_original_resolution = annotation_stats.dataset[dataset][
        'video_id_to_original_resolution']
    # make track ID unique across different videos
    track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
        annotations)
    video_tile_to_uniq_track_id = collections.defaultdict(list)

    for track_id, track_annotations in track_annotations_grp:
        videoid = track_annotations.iloc[0]['videoid']
        image_resolution = video_id_to_original_resolution[videoid]

        for _, row in track_annotations.iterrows():
            bbox = _clamp_bbox(image_resolution,
                               (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
            tile_coords = _get_tile_coords_from_bbox(image_resolution, bbox,
                                                     long_edge_ratio=0.5, short_edge_ratio=1)
            for tile_coord in tile_coords:
                tile_id = _combine_imageid(row['videoid'], row['frameid'], *tile_coord)
                video_tile_to_uniq_track_id[tile_id].append(track_id)

    # print(video_tile_to_uniq_track_id)
    all_unique_track_ids = set(track_annotations_grp.groups.keys())
    print("Parsed annotations. Found {} unique track IDs in {}.".format(
        len(all_unique_track_ids), ','.join(all_unique_track_ids)))
    return all_unique_track_ids, video_tile_to_uniq_track_id


# def get_video_frame_to_uniq_track_id(base_dir, dataset):
#     load_annotation_func = annotation_stats.dataset[dataset][
#         'annotation_func']
#     labels = annotation_stats.dataset[dataset]['labels']
#     annotation_dir = os.path.join(base_dir, datasets[dataset][0])
#     annotations = load_annotation_func(annotation_dir)
#     test_video_ids = annotation_stats.dataset[dataset]['test']
#     annotations = annotations[annotations['videoid'].isin(test_video_ids)]
#     annotations = annotation.filter_annotation_by_label(
#         annotations, labels=labels)
#     # make track ID unique across different videos
#     track_annotations_grp = annotation.group_annotation_by_unique_track_ids(
#         annotations)
#     video_frame_to_uniq_track_id = collections.defaultdict(list)
#     for track_id, track_annotations in track_annotations_grp:
#         for _, row in track_annotations.iterrows():
#             video_id = row['videoid']
#             frame_id = row['frameid']
#             video_frame_to_uniq_track_id[(video_id, frame_id)].append(track_id)
#     all_unique_trakc_ids = set(track_annotations_grp.groups.keys())
#     print("Parsed annotations. Found {} unique track IDs in {}.".format(
#         len(all_unique_trakc_ids), ','.join(all_unique_trakc_ids)))
#     return all_unique_trakc_ids, video_frame_to_uniq_track_id



if __name__ == '__main__':
    fire.Fire()
