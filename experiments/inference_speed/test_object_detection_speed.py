import glob
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import time

import tensorflow as tf
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)
sys.path.append("..")

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_inception_v2_coco_2017_11_08'
# MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'


MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
tf.logging.info('Using model: {}'.format(MODEL_NAME))

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

dir_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(dir_path, MODEL_NAME)):
    tf.logging.info('Downloading {} --> {}'.format(DOWNLOAD_BASE, MODEL_FILE))
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(
        np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'sample-images'
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*'))
# PATH_TO_TEST_IMAGES_DIR = 'test_images'
# TEST_IMAGE_PATHS = [
#     os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i))
#     for i in range(1, 3)
# ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

latencies = []
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
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

        # warm up
        image_np = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
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
            for image_path in TEST_IMAGE_PATHS:
                tf.logging.info(image_path)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
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
