from __future__ import absolute_import, division, print_function

import argparse
import glob
import numpy as np
import time
import os

import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    return input_image, normalized


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
        np.uint8)


def _divide_to_tiles(im, grid_w, grid_h, tile_w, tile_h, allocated_tiles):
    # assert (im_w % grid_w == 0), \
    #     "image width ({}) cannot be evenly divided to ({}) pieces".format(
    #         im_w, grid_w)
    # assert (im_h % grid_h == 0), \
    #     "image height ({}) cannot be evenly divided to ({}) pieces".format(
    #        im_h, grid_h)

    # row majored. if tiles are divided into 2x2
    # then the sequence is (0,0), (0,1), (1,0), (1,1)
    # in which 1st index is on x-aix, 2nd index on y-axis
    for h_idx in range(0, grid_w):
        for v_idx in range(0, grid_h):
            tile_x = int(h_idx * tile_w)
            tile_y = int(v_idx * tile_h)
            current_tile = im[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
            tiles[h_idx * grid_h + v_idx] = current_tile
    return tiles


def _tf_divide_to_tiles(images_tensor, tile_w, tile_h, grid_w, grid_h):
    tiles = []
    for h_idx in range(0, grid_w):
        for v_idx in range(0, grid_h):
            tile_x = h_idx * tile_w
            tile_y = v_idx * tile_h
            current_tile = tf.image.crop_to_bounding_box(
                images_tensor, tile_y, tile_x, tile_h, tile_w)
            tiles.append(current_tile)
    tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])
    tf.logging.info('tile shape before concat: {}'.format(tiles))
    tiles = tf.reshape(tiles, [-1, tile_h, tile_w, 3])
    tf.logging.info('tile shape after reshape: {}'.format(tiles))
    return tiles


if __name__ == "__main__":
    model_file = None
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "MobilenetV1/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--image_dir", help="Test image directory")
    parser.add_argument("--grid_w", help="# of tile horizontally")
    parser.add_argument("--grid_h", help="# of tiles vertically")
    parser.add_argument("--im_w", help="# of tile horizontally")
    parser.add_argument("--im_h", help="# of tiles vertically")
    parser.add_argument(
        "--batch_size", help="max # of tiles to feed in at once")

    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    assert args.image_dir
    image_dir = args.image_dir
    assert args.grid_w
    grid_w = int(args.grid_w)
    assert args.grid_h
    grid_h = int(args.grid_h)
    assert args.im_w
    im_w = int(args.im_w)
    assert args.im_h
    im_h = int(args.im_h)
    assert args.batch_size
    batch_size = int(args.batch_size)

    graph = load_graph(model_file)

    with graph.as_default():
        input_image = tf.placeholder(tf.uint8, shape=[None, None, None, 3])
        resized = tf.image.resize_images(
            input_image, [input_height * grid_h, input_width * grid_w],
            method=tf.image.ResizeMethod.BILINEAR)
        float_caster = tf.cast(resized, tf.float32)
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    test_images_path = glob.glob(os.path.join(image_dir, '*'))
    tf.logging.info('model_file: {}'.format(model_file))

    latencies = []
    normalization_latencies = []
    tile_normalization_latencies = []

    tiles = np.zeros(shape=(grid_w * grid_h, input_height, input_width, 3))
    # use numpy for tiling
    with tf.Session(graph=graph) as sess:
        tf.logging.info('Start session')

        # warm up
        image_np = (np.random.rand(batch_size, input_height, input_width, 3) *
                    255).astype(np.uint8)
        normalized_image = sess.run(normalized, {input_image: image_np})
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: normalized_image
        })
        results = np.squeeze(results)
        tf.logging.info('Finished warming up.')

        for _ in range(3):
            for image_path in test_images_path:
                tf.logging.info(image_path)
                image = Image.open(image_path)
                image_np = np.asarray(image)

                st = time.time()
                normalized_image = sess.run(normalized, {input_image: tiles})
                normalization_latencies.append(time.time() - st)

                tiles = _divide_to_tiles(
                    normalized_image,
                    grid_w, grid_h,
                    input_width,
                    input_height,
                    tiles)
                tf.logging.info('GOt {} tiles.'.format(len(tiles)))
                tile_normalization_latencies.append(time.time() - st)

                processed_tiles_num = 0
                while processed_tiles_num < grid_h * grid_w:
                    batch_tiles = tiles[
                        processed_tiles_num: processed_tiles_num + batch_size]
                    results = sess.run(output_operation.outputs[0], {
                        input_operation.outputs[0]: normalized_image
                    })
                    results = np.squeeze(results)
                    processed_tiles_num += batch_size

                latencies.append(time.time() - st)

    tf.logging.info('average latency: {:.1f}ms, std: {:.1f}ms'.format(
        np.mean(latencies) * 1000,
        np.std(latencies) * 1000))
    tf.logging.info('normalization latency: {:.1f}ms, std: {:.1f}ms'.format(
        np.mean(normalization_latencies) * 1000,
        np.std(normalization_latencies) * 1000))
    tf.logging.info(
        'tile + normalization latency: {:.1f}ms, std: {:.1f}ms'.format(
            np.mean(tile_normalization_latencies) * 1000,
            np.std(tile_normalization_latencies) * 1000))
    tf.logging.info('latencies: {}'.format(latencies))
