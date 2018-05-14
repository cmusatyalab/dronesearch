import os
import sys
import tarfile
import time

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)
sys.path.append("..")


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
        np.uint8)


models = {
    'ssd_mobilenet': 'ssd_mobilenet_v1_coco_2017_11_17',
    'ssd_inceptionv2': 'ssd_inception_v2_coco_2017_11_17',
    'faster_rcnn_inceptionv2': 'faster_rcnn_inception_v2_coco_2017_11_08',
    'faster_rcnn_resnet101': 'faster_rcnn_resnet101_coco_2017_11_08'
}


def load_graph(model_name):
    tf_model_name = models[model_name]
    model_file = tf_model_name + '.tar.gz'
    download_base_url = 'http://download.tensorflow.org/models/object_detection/'
    tf.logging.info('Using model: {}'.format(tf_model_name))

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    path_to_ckpt = tf_model_name + '/frozen_inference_graph.pb'

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(dir_path, tf_model_name)):
        tf.logging.info('Downloading {} --> {}'.format(download_base_url, model_file))
        opener = urllib.request.URLopener()
        opener.retrieve(download_base_url + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def create_samples(num_sample_per_run, im_h, im_w):
    return (np.random.randn(num_sample_per_run, im_h, im_w, 3) * 255).astype(np.uint8)


def get_io_tensors(detection_graph):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections


def main(model_name, input_height=224, input_width=224, num_sample_per_run=100, num_run=3):
    detection_graph = load_graph(model_name)
    latencies = []
    samples = create_samples(num_sample_per_run, input_height, input_width)
    # needed for jetson to be able to allocate enough memory
    # see https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = get_io_tensors(
                detection_graph)
            # warm up
            image_np = (np.random.rand(input_height, input_width, 3) * 255).astype(np.uint8)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [
                    detection_boxes, detection_scores, detection_classes,
                    num_detections
                ],
                feed_dict={
                    image_tensor: image_np_expanded
                })

            for _ in range(3):
                for image in samples:
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image, axis=0)
                    # Actual detection.
                    st = time.time()
                    (boxes, scores, classes, num) = sess.run(
                        [
                            detection_boxes, detection_scores, detection_classes,
                            num_detections
                        ],
                        feed_dict={
                            image_tensor: image_np_expanded
                        })
                    latencies.append(time.time() - st)

    tf.logging.info('average latency: {:.1f}ms, std: {:.1f}ms'.format(
        np.mean(latencies) * 1000, np.std(latencies) * 1000))
    tf.logging.info('latencies: {}'.format(latencies))


if __name__ == '__main__':
    main()
