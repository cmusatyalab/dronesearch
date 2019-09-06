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
# video_ids=(
#     "01"
#     "02"
#     "03"
#     "04"
#     "05"
#     "06"
#     "07"
#     "08"
#     "09"
#     "10"
#     "11"
# )
video_ids=(
    '1.1.3'
    '1.1.2'
    '1.1.5'
    '1.1.4'
    '2.2.7'
    '2.1.7'
    '2.1.4'
    '2.2.11'
    '1.1.10'
    '2.2.2'
    '2.2.4'
    '1.1.7'
)
dataset_dir=$1
long_edge=$2
short_edge=$3
for video_id in "${video_ids[@]}"
do
    echo ${video_id}
    python preprocess.py resize-frame-sequence-by-id ${dataset_dir}/images ${dataset_dir}/images_${long_edge}_${short_edge} "$video_id" ${long_edge} ${short_edge} &
done
wait
