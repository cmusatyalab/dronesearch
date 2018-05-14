#!/usr/bin/env bash

read -p "Please Enter Device Name: " device

if [[ device == "jetson" ]];then
    sudo systemctl stop fwupd.service
    sudo systemctl stop snapd.service
fi

printf "launching MobileNet test"
python test_classification_speed.py \
--model-file=${HOME}/mobisys18/pretrained_models/frozen_mobilenet_v1.pb \
--input_mean=128 --input_std=128 --output-layer=MobilenetV1/Predictions/Reshape_1 2>&1 \
| tee results/tf_mobilenetv1_${device}_no_preprocessing.txt

printf "launching ResNet test"
python test_classification_speed.py \
--model-file=${HOME}/mobisys18/pretrained_models/frozen_resnet_v1_101.pb \
--output-layer=resnet_v1_101/predictions/Reshape_1 2>&1 \
| tee results/tf_resnet101_${device}_no_preprocessing.txt
