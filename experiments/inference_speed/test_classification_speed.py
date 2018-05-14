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

import time

import fire
import numpy as np
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


def create_samples(num_sample_per_run, im_h, im_w):
    return (np.random.randn(num_sample_per_run, im_h, im_w, 3) * 255).astype(np.uint8)


def main(model_file, output_layer, input_height=224, input_width=224, input_mean=0, input_std=256,
         input_layer="input", num_run=3, num_sample_per_run=100):
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

    tf.logging.info('---------------------------------------------------------------------------')
    tf.logging.info('-------Running Experiements------------------------------------------------')
    tf.logging.info('-------Batch 1. Forward Pass only. Preprocessing Excluded------------------')
    tf.logging.info('model_file: {}'.format(model_file))

    latencies = []
    samples = create_samples(num_sample_per_run, input_height, input_width)
    # needed for jetson to be able to allocate enough memory
    # see https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        # warm up
        image_np = (np.random.rand(input_height, input_width, 3) * 255).astype(np.uint8)
        normalized_image = sess.run(normalized, {
            input_image: image_np
        })
        _ = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: normalized_image
        })

        for _ in range(num_run):
            for image_np in samples:
                normalized_image = sess.run(normalized, {
                    input_image: image_np
                })
                st = time.time()
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: normalized_image
                })
                latencies.append(time.time() - st)
    tf.logging.info('average latency: {:.1f}ms, std: {:.1f}ms'.format(
        np.mean(latencies) * 1000, np.std(latencies) * 1000))
    tf.logging.info('latencies: {}'.format(latencies))


if __name__ == "__main__":
    fire.Fire(main)
