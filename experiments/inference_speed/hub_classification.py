from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from logzero import logger

test_models = {
    'mobilenet':
        'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1',
    'resnet101': 'https://tfhub.dev/google/imagenet/resnet_v1_101/classification/1'
}


def run_experiment(module_url, sample_num=100, run_num=3):
    times = []
    with tf.Graph().as_default():
        module = hub.Module(module_url)
        height, width = hub.get_expected_image_size(module)
        logger.info("model input should be ?x{}x{}".format(width, height))
        input = tf.placeholder(shape=(None, height, width, 3), dtype=tf.float32)
        output = module(input)
        samples = np.random.rand(sample_num, height, width, 3) * 255
        with tf.train.MonitoredSession() as sess:
            # warm-up
            logger.debug("warming up")
            for warmup_run in range(3):
                image = samples[0]
                image = np.expand_dims(image, axis=0)
                sess.run(output, feed_dict={input: image})

            logger.debug("Warm up finished. Start running.")
            for run in range(run_num):
                for image in samples:
                    image = np.expand_dims(image, axis=0)
                    st = time.time()
                    [_] = sess.run(output, feed_dict={input: image})
                    end = time.time()
                    times.append((end - st) * 1000)
    times = np.array(times)
    logger.info("Tested with {} images for {} runs. average time is {:.1f} ms, std is {:.1f} ms".format(sample_num,
                                                                                                        run_num,
                                                                                                        np.mean(times),
                                                                                                        np.std(times)))


if __name__ == '__main__':
    for k, v in test_models.items():
        logger.info("\n\n=====================================")
        logger.info("Running experiment for {}".format(k))
        run_experiment(v)
