#!/usr/bin/env bash -ex

printf "--------------------------------------------"
printf "-----running classification experiment------"
classification_models=("mobilenet resnet101")
printf "classification models include: ${classification_models[@]}"

for model in ${classification_models[@]};
do
    if [[ model == "mobilenet" ]]; then
        input_mean=128
        input_std=128
        output_layer="MobilenetV1/Predictions/Reshape_1"
    else
        input_mean=0
        input_std=256
    fi
    python test_classification_speed.py --graph=/home/junjuew/mobisys18/pretrained_models/inception_v2_2016_08_28_frozen.pb 2>&1 | tee ~/mobisys18/experiments/inference_speed/tf_inceptionv2_k40.txt
done
