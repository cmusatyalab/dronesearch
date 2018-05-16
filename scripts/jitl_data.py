import pickle
from operator import itemgetter

import fire
import glob
import cv2

import itertools
import numpy as np
import os
from StringIO import StringIO

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import PIL.Image

from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd

from annotation_stats import dataset as dataset_stats
from io_util import load_all_pickles_from_dir

np.random.seed(42)

datasets = {
    'elephant':
        ('elephant/annotations',
         'elephant/classification_448_224_224_224_annotations',
         'elephant/experiments/classification_448_224_224_224_extra_negative/test_inference'
         ),
    'raft':
        ('raft/annotations', 'raft/classification_448_224_224_224_annotations',
         'raft/experiments/classification_448_224_224_224_extra_negative/test_inference'
         ),
    'okutama':
        ('okutama/annotations',
         'okutama/classification_448_224_224_224_annotations',
         'okutama/experiments/classification_448_224_224_224_extra_negative/test_inference'
         ),
    'stanford':
        ('stanford/annotations',
         'stanford/classification_448_224_224_224_annotations',
         'stanford/experiments/classification_448_224_224_224_extra_negative/test_inference'
         ),
}
dataset_to_results = datasets


def _get_videoid(image_id):
    return _split_imageid(image_id)[0]


def _split_imageid(image_id):
    tokens = image_id.split('_')
    video_id, frame_id, grid_x, grid_y = '_'.join(tokens[:-3]), tokens[-3], tokens[-2], tokens[-1]
    frame_id = int(frame_id)
    grid_x = int(grid_x)
    grid_y = int(grid_y)
    return video_id, frame_id, grid_x, grid_y


def _combine_imageid(*args):
    return '_'.join(map(str, args))


def _increment_frame_id(image_id, decrement=False):
    video_id, frame_id, grid_x, grid_y = _split_imageid(image_id)
    if not decrement:
        return _combine_imageid(video_id, frame_id + 1, grid_x, grid_y)
    else:
        return _combine_imageid(video_id, frame_id - 1, grid_x, grid_y)


def _get_tile(image_id, dataset_dir, tile_width, tile_height):
    # image_dir = os.path.join(dataset_dir, 'photos')
    # assert os.path.exists(image_dir), "{} doesn't exist".format(image_dir)
    image_dir = dataset_dir

    # print("getting tile for", image_id)
    tokens = image_id.split('_')
    video_id, frame_id, grid_x, grid_y = '_'.join(tokens[:-3]), tokens[-3], tokens[-2], tokens[-1]
    frame_id = int(frame_id)
    grid_x = int(grid_x)
    grid_y = int(grid_y)
    base_image_path = os.path.join(image_dir, video_id, '{:010d}'.format(
        int(frame_id + 1))) + '.jpg'
    print("reading image from: ", base_image_path)
    im = cv2.imread(base_image_path)
    if im is None:
        raise ValueError('Failed to load image: '.format(base_image_path))
    tile_x = grid_x * tile_width
    tile_y = grid_y * tile_height
    # print(im.shape, tile_x, tile_y, tile_width, tile_height)
    current_tile = im[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
    ret, encoded_tile = cv2.imencode('.jpg', current_tile)
    if not ret:
        raise ValueError('Failed to encode tile: '.format(image_id))
    return encoded_tile.tobytes()


def visualize_tp_fp(tile_classification_annotation_file,
                    tile_test_inference_file,
                    image_dir,
                    n_samples=5,
                    tile_width=224,
                    tile_height=224):
    ground_truth, _, predictions, sorted_imageids = _parse_tile_annotation_and_inference_pre_logit(
        tile_classification_annotation_file, tile_test_inference_file)
    fp_indexes = np.nonzero(np.logical_and(ground_truth == 0, predictions == 1))[0]
    tp_indexes = np.nonzero(np.logical_and(ground_truth == 1, predictions == 1))[0]
    # only take n_samples from each
    fp_sample_index = resample(fp_indexes, n_samples=n_samples)
    tp_sample_index = resample(tp_indexes, n_samples=n_samples)

    print(fp_sample_index, tp_sample_index)

    # TODO check if use of itemgetter is buggy when index is singleton
    fp_imageids = itemgetter(*fp_sample_index)(sorted_imageids)
    tp_imageids = itemgetter(*tp_sample_index)(sorted_imageids)

    # plt.figure(figsize=(tile_width * 2.1, tile_height*n_samples*1.1))
    f, axarr = plt.subplots(2, n_samples)
    f.tight_layout()
    f.subplots_adjust(hspace=0.1, wspace=0.05)
    for row, index in zip([0, 1], [fp_imageids, tp_imageids]):
        for col in range(n_samples):
            imageid = index[col]
            encoded_tile = _get_tile(_increment_frame_id(imageid, decrement=True),
                                     image_dir,
                                     tile_width=tile_width,
                                     tile_height=tile_height)
            axarr[row, col].axis('off')
            axarr[row, col].imshow(PIL.Image.open(StringIO(encoded_tile)))

    plt.show()


def make_jitl_dataframe(dataset,
                        base_dir,
                        output_file=None
                        ):
    """
    Generate a per-dataset dataframe containing necessary data fro JITL experiments.
    Columns: label, feature, prediction_proba, imageid.
    label: ground truth 0/1
    feature: [int(1024)]
    prediction_proba: [float(2)] DNN's prediction score
    imageid: "<video_id>_<frame_id>_<grid_x>_<grid_y>"
    :param base_dir: where to find the input files using paths in results.
    :param dataset: dataset name. e.g, 'stanford', 'raft'
    :param output_file:
    :return:
    """

    df = _parse_tile_annotation_and_inference_pre_logit(dataset,
                                                        os.path.join(base_dir, dataset_to_results[dataset][1]),
                                                        os.path.join(base_dir, dataset_to_results[dataset][2]),
                                                        interested_videoids=dataset_stats[dataset]['test'])

    print("Confusion matrix (using 0.5 cutoff)")
    ground_truth = np.array(df['label'])
    # prediction_proba=np.array(df['prediction_proba'].tolist())
    # print prediction_proba[:5]
    predictions = np.argmax(np.array(df['prediction_proba'].tolist()), axis=1)
    assert ground_truth.shape == predictions.shape, "{}, {}".format(ground_truth.shape, predictions.shape)
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    print(str(cm))

    print("Sample 10 rows.")
    print df.iloc[::df.shape[0] / 10]

    if output_file:
        df.to_pickle(output_file)


def _parse_tile_annotation_and_inference_pre_logit(dataset,
                                                   tile_classification_annotation_dir,
                                                   tile_test_inference_dir,
                                                   interested_videoids,
                                                   extra_negative=True):
    print("Interested video IDs: {}".format(','.join(interested_videoids)))
    tile_ground_truth = load_all_pickles_from_dir(tile_classification_annotation_dir)
    print("Loaded {} results from  annotation: {} ".format(len(tile_ground_truth), tile_classification_annotation_dir))
    # Filter interested video ids and transform true/false to 1/0
    tile_ground_truth = dict([(k, 1 if v else 0) for k, v in tile_ground_truth.iteritems()
                              if _get_videoid(k) in interested_videoids])
    print("Filter {} for interested videos".format(len(tile_ground_truth)))

    tile_inference_result_and_pre_logit = load_all_pickles_from_dir(tile_test_inference_dir,
                                                                    prefix=dataset if extra_negative else '')
    print(
        "Loaded {} results from inference: {}".format(len(tile_inference_result_and_pre_logit),
                                                      tile_test_inference_dir))

    # filter image ids in interested videos and
    # XXX bring down 1-off frame ids!
    if not extra_negative:
        tile_inference_result_and_pre_logit = dict((_increment_frame_id(k, decrement=True), v)
                                                   for k, v in tile_inference_result_and_pre_logit.iteritems()
                                                   if _get_videoid(k) in interested_videoids)
    else:
        tile_inference_result_and_pre_logit = dict((_increment_frame_id(k.split('/', 1)[1], decrement=True), v)
                                                   for k, v in tile_inference_result_and_pre_logit.iteritems()
                                                   if k.startswith(dataset + '/') and _get_videoid(
            k.split('/', 1)[1]) in interested_videoids)
    print("Filter {} for interested videos".format(len(tile_inference_result_and_pre_logit)))

    imageids = tile_ground_truth.keys()
    assert set(imageids) == set(tile_inference_result_and_pre_logit.keys()), "Probably due to 1-off imageids?"
    # assert set(imageids).issubset(set(tile_inference_result_and_pre_logit.keys())), "Probably due to 1-off image ids?"

    sorted_imageids = sorted(imageids, key=_split_imageid)  # hopefully sorts by timestamps
    ground_truth = np.array([tile_ground_truth[imgid] for imgid in sorted_imageids])
    prediction_proba = np.array([tile_inference_result_and_pre_logit[imgid][:2] for imgid in sorted_imageids])
    pre_logit = np.stack([np.array(tile_inference_result_and_pre_logit[imgid][2:]) for imgid in sorted_imageids])
    assert len(sorted_imageids) == ground_truth.shape[0] == prediction_proba.shape[0]
    assert prediction_proba.shape[1] == 2
    assert pre_logit.shape[1] == 1024, prediction_proba.shape

    df = pd.DataFrame({'label': ground_truth.tolist(),
                       'feature': pre_logit.tolist(),
                       'prediction_proba': prediction_proba.tolist(),
                       'imageid': sorted_imageids})

    print(df.iloc[:5])
    return df


def make_jitl_dnn_threshold(early_discard_plotted_threshold_and_recall_file, dataset, output_file):
    """
    Get DNN thresholds that event recall changes
    :param early_discard_plotted_threshold_and_recall_file: The DNN threshold vs event recall pkl file that contains
    the points used in the early discard plot
    :param dataset: dataset name
    :param output_file: Outputfile that contains a numpy array of DNN threshold in increasing order
    :return:
    """
    plotted_data = np.load(early_discard_plotted_threshold_and_recall_file)
    assert dataset in plotted_data, "Dataset name {} is not in the plotted threshold vs recall data file"
    interesting_dnn_threshold = np.array(plotted_data[dataset][0])
    interesting_dnn_threshold.sort()
    with open(output_file, 'wb') as f:
        np.save(f, interesting_dnn_threshold)
    print("interesting DNN threshold for {}:\n{}".format(dataset, interesting_dnn_threshold))


if __name__ == '__main__':
    fire.Fire()
