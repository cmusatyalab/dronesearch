#
# Drone Search
#
# A computer vision pipeline for live video search on drone video
# feeds leveraging edge servers.
#
# Copyright (C) 2018-2019 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SCENE_FRAME_SEQUENCE_DIR="frames/nexus"
# SCENE_ANNOTATION_DIR="annotations_car_only/nexus"
# TRAIN_LIST='["video1","video2","video5","video8","video9","video11"]'
# OUTPUT_DIR="experiments/car_only_0"
# # test list is "video4","video7"

# python gen_experiments.py gen_car_only_experiments $SCENE_FRAME_SEQUENCE_DIR $SCENE_ANNOTATION_DIR $TRAIN_LIST $OUTPUT_DIR

BASE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus"

SCENE_FRAME_SEQUENCE_DIR="${BASE_DIR}/frames/nexus"
SCENE_ANNOTATION_DIR="${BASE_DIR}/annotations_car_only/nexus"
TRAIN_LIST='["video4","video7"]'
OUTPUT_DIR="${BASE_DIR}/experiments/car_only_1/validation"
# test list is "video4","video7"

python preprocess_for_tpod_faster_rcnn.py gen_car_only_experiments $SCENE_FRAME_SEQUENCE_DIR $SCENE_ANNOTATION_DIR $TRAIN_LIST $OUTPUT_DIR
