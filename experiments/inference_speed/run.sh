#!/usr/bin/env bash

read -p "Please Enter Device Name" device

printf "launching MobileNet test"
python test_classification_speed.py \
--model-file=${HOME}/mobisys18/pretrained_models/frozen_mobilenet_v1.pb \
--input_mean=128 --input_std=128 --output-layer=MobilenetV1/Predictions/Reshape_1 2>&1 \
| tee results/tf_mobilenetv1_${device}.txt

printf "launching ResNet test"
python test_classification_speed.py \
--model-file=${HOME}/mobisys18/pretrained_models/frozen_resnet_v1_101.pb \
--output-layer=resnet_v1_101/predictions/Reshape_1 2>&1 \
| tee ~/mobisys18/experiments/inference_speed/tf_resnet_v1_101_${device}.txt
