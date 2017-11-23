# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import argparse
import glob
import numpy as np
import sys
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
    input_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
    float_caster = tf.cast(input_image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander,
                                       [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    return input_image, normalized


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
        np.uint8)


if __name__ == "__main__":
    model_file = None
    label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    input_layer = "input"
    output_layer = "InceptionV2/Predictions/Reshape_1"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
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

    graph = load_graph(model_file)
    with graph.as_default():
        input_image, normalized = read_tensor_from_image_file(
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    PATH_TO_TEST_IMAGES_DIR = 'sample-images'
    TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,
                                              '*'))
    tf.logging.info('model_file: {}'.format(model_file))

    latencies = []
    with tf.Session(graph=graph) as sess:
        # warm up
        image_np = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        normalized_image = sess.run(normalized, {
            input_image: image_np
        })
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: normalized_image
        })
        results = np.squeeze(results)

        for _ in range(3):
            for image_path in TEST_IMAGE_PATHS:
                tf.logging.info(image_path)
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                # Actual detection.
                st = time.time()
                normalized_image = sess.run(normalized, {
                    input_image: image_np
                })
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: normalized_image
                })
                results = np.squeeze(results)
                latencies.append(time.time() - st)
    tf.logging.info('average latency: {:.1f}ms, std: {:.1f}ms'.format(
        np.mean(latencies) * 1000, np.std(latencies) * 1000))
    tf.logging.info('latencies: {}'.format(latencies))
