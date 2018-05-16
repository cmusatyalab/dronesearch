#!/usr/bin/env bash
# This script creates frozen graph into pretrained_models

pretrain_model_dir=/home/junjuew/mobisys18/pretrained_models
bazel build tensorflow/python/tools:freeze_graph
bazel build tensorflow/tools/graph_transforms:summarize_graph

cd /home/junjuew/tensorflow/models/research/slim
model_name=resnet_v1_101
output_node_name=resnet_v1_101/predictions/Reshape_1
label_offset=1 # used by VGG and ResNet since they do not include background class for imagenet
python export_inference_graph.py --alsologtostderr \
    --model_name=${model_name} \
    --labels_offset=${label_offset} \
    --output_file=${pretrain_model_dir}/${model_name}.pb

# used to get the output_node_names
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=${pretrain_model_dir}/${model_name}.pb

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${pretrain_model_dir}/${model_name}.pb \
  --input_checkpoint=${pretrain_model_dir}/${model_name}.ckpt \
  --input_binary=true --output_graph=${pretrain_model_dir}/frozen_${model_name}.pb \
  --output_node_names=${output_node_name}
