#!/usr/bin/env bash

read -p "Please Enter Device Name: " device

if [[ device == "jetson" ]];then
    sudo systemctl stop fwupd.service
    sudo systemctl stop snapd.service
fi

#printff "========================Running Classification Test====================="
#printf "launching MobileNet test"
#python test_classification_speed.py \
#--model-file=${HOME}/mobisys18/pretrained_models/frozen_mobilenet_v1.pb \
#--input_mean=128 --input_std=128 --output-layer=MobilenetV1/Predictions/Reshape_1 2>&1 \
#| tee results/tf_mobilenetv1_${device}_no_preprocessing.txt
#
#printf "launching ResNet test"
#python test_classification_speed.py \
#--model-file=${HOME}/mobisys18/pretrained_models/frozen_resnet_v1_101.pb \
#--output-layer=resnet_v1_101/predictions/Reshape_1 2>&1 \
#| tee results/tf_resnet101_${device}_no_preprocessing.txt


printf "========================Running Detection Test=====================\n"
models=(
    'ssd_mobilenet'
#    'ssd_inceptionv2'
#    'faster_rcnn_inceptionv2'
#    'faster_rcnn_resnet101'
)
for model_name in ${models[@]};do
    printf "launching detection test for ${model_name}"
    python test_object_detection_speed.py ${model_name} 2>&1 | tee \
    results/tf_${model_name}_${device}_no_preprocessing_cpu.txt
done
