package org.tensorflow.demo;

import java.util.HashMap;
import java.util.Map;

public class ExperimentConfig {
    public static class ModelConfig {
        public enum Type {
            CLASSIFIER, DETECTOR
        }

        public int input_size;
        public int image_mean;
        public float image_std;
        public Type type;
        public String input_name;
        public String output_name;
        public String model_file;
        public String label_file;
        public String image_dir;

        public ModelConfig(int input_size, int image_mean, float image_std, Type type, String
                input_name, String output_name, String model_file, String label_file,
                           String image_dir) {
            this.input_size = input_size;
            this.image_mean = image_mean;
            this.image_std = image_std;
            this.type = type;
            this.input_name = input_name;
            this.output_name = output_name;
            this.model_file = model_file;
            this.label_file = label_file;
            this.image_dir = image_dir;
        }
    }

    public static final Map<String, ModelConfig> configs;

    static {
        configs = new HashMap<>();
/*        configs.put("mobilenet", new ModelConfig(
                224,
                128,
                128,
                ModelConfig.Type.CLASSIFIER,
                "input",
                "MobilenetV1/Predictions/Reshape_1",
                "file:///android_asset/frozen_mobilenet_v1.pb",
                "file:///android_asset/imagenet_comp_graph_label_strings.txt",
                "android_samples" // resolution 224x224
        ));
        configs.put("resnet101",
                new ModelConfig(
                        224,
                        0,
                        255,
                        ModelConfig.Type.CLASSIFIER,
                        "input",
                        "resnet_v1_101/predictions/Reshape_1",
                        "file:///android_asset/frozen_resnet_v1_101.pb",
                        "file:///android_asset/imagenet_comp_graph_label_strings.txt",
                "android_samples" // resolution 224x224
                ));*/
        configs.put("ssd_mobilenet",
                new ModelConfig(
                        224,
                        -1,
                        -1,
                        ModelConfig.Type.DETECTOR,
                        null,
                        null,
                        "file:///android_asset/ssd_mobilenet_v1_android_export.pb",
                        "file:///android_asset/imagenet_comp_graph_label_strings.txt",
                        "android_samples_300x300" // resolution 300x300
                ));
        configs.put("ssd_inception",
                new ModelConfig(
                        224,
                        -1,
                        -1,
                        ModelConfig.Type.DETECTOR,
                        null,
                        null,
                        "file:///android_asset/ssd_inception_v2.pb",
                        "file:///android_asset/imagenet_comp_graph_label_strings.txt",
                        "android_samples_300x300" // resolution 300x300
                ));
/*        configs.put("faster_rcnn_inception",
                new ModelConfig(
                        224,
                        -1,
                        -1,
                        ModelConfig.Type.DETECTOR,
                        null,
                        null,
                        "file:///android_asset/faster_rcnn_inception_v2.pb",
                        "file:///android_asset/imagenet_comp_graph_label_strings.txt"
                ));
        configs.put("faster_rcnn_resnet101",
                new ModelConfig(
                        224,
                        -1,
                        -1,
                        ModelConfig.Type.DETECTOR,
                        null,
                        null,
                        "file:///android_asset/faster_rcnn_resnet101_v1.pb",
                        "file:///android_asset/imagenet_comp_graph_label_strings.txt"
                ));*/
    }
}
