import glob
import os

import fire
import numpy as np
import tensorflow as tf
from PIL import Image
from logzero import logger
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


class TFModel(object):
    def __init__(self, frozen_graph_path, label_file_path, num_classes):
        self._frozen_graph_path = frozen_graph_path
        self._label_file_path = label_file_path
        self._num_classes = num_classes
        self._graph = self._load_graph(self._frozen_graph_path)
        self._category_index = self._load_label_map(self._label_file_path, self._num_classes)
        self._output_tensor_dict = self._get_inference_output_tensors(
            self._graph,
            ['num_detections', 'detection_boxes',
             'detection_scores', 'detection_classes',
             'detection_masks'])
        self._sess = tf.Session(graph=self._graph)

    @property
    def category_index(self):
        return self._category_index

    @staticmethod
    def _load_graph(frozen_graph_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    @staticmethod
    def _load_label_map(label_file_path, num_classes):
        label_map = label_map_util.load_labelmap(label_file_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    @staticmethod
    def _get_inference_output_tensors(graph, tensor_list):
        tensor_dict = {}
        with graph.as_default():
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            for key in tensor_list:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
        return tensor_dict

    def run_batch_inference(self, images):
        """Run inference on a single image. Not compatible with Mask-RCNN

        :param graph: Tensorflow graph with weights loaded
        :param output_tensor_dict: Dictionary of output tensor names to tensors
        :return: Inference results as a dictionary
        """
        if len(images.shape) == 3:
            logger.info("input is a single image. Use batch size 1.")
            images = np.expand_dims(images, axis=0)
        output_tensor_dict = self._output_tensor_dict
        image_tensor = self._graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self._sess.run(output_tensor_dict,
                                     feed_dict={image_tensor: images})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = output_dict['num_detections'].astype(np.uint)
        output_dict['detection_classes'] = output_dict[
            'detection_classes'].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes']
        output_dict['detection_scores'] = output_dict['detection_scores']
        return output_dict

    def close(self):
        self._sess.close()
        print("tensorflow session closed.")


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_on_test_dir(frozen_graph_path, label_file_path, num_classes, test_dir, output_dir):
    tfmodel = TFModel(frozen_graph_path, label_file_path, num_classes)
    test_image_paths = glob.glob(os.path.join(test_dir, '*'))
    for image_path in test_image_paths:
        image = Image.open(image_path)
        # # the array based representation of the image will be used later in order to prepare the
        # # result image with boxes and labels on it.
        img = load_image_into_numpy_array(image)
        # Actual detection.
        output_dict = tfmodel.run_inference_for_single_image(img)
        # Visualization of the results of a detection.
        img = visualize_highest_prediction_per_class(img, output_dict, tfmodel.category_index)
        annotated_image = Image.fromarray(img)
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        annotated_image.save(output_image_path)
        logger.info('{} --> {}'.format(image_path, output_image_path))


def visualize_highest_prediction_per_class(img, output_dict, category_index, min_score_thresh=0.5,
                                           num_predictions_per_class=10):
    # find the highest scored boxes
    detection_boxes = []
    detection_classes = []
    detection_scores = []
    detected_classes = set(output_dict['detection_classes'].tolist())
    # find the highest scored box by class
    for cls_idx in detected_classes:
        idx_in_detection_array = np.where(output_dict['detection_classes'] == cls_idx)[0]
        # tensorflow output is already sorted by scores. choose top num_predictions_per_class
        sorted_idx_by_score = idx_in_detection_array
        top_idx_by_score = np.array(sorted_idx_by_score[:num_predictions_per_class])
        cls_detection_boxes = output_dict['detection_boxes'][top_idx_by_score]
        cls_detection_classes = output_dict['detection_classes'][top_idx_by_score]
        cls_detection_scores = output_dict['detection_scores'][top_idx_by_score]
        detection_boxes.extend(cls_detection_boxes)
        detection_classes.extend(cls_detection_classes)
        detection_scores.extend(cls_detection_scores)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.array(detection_boxes),
        np.array(detection_classes),
        np.array(detection_scores),
        category_index,
        max_boxes_to_draw=len(detection_boxes),
        min_score_thresh=min_score_thresh,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    return img


if __name__ == '__main__':
    fire.Fire()
