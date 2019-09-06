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
BASE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus"
IMAGE_DIR="${BASE_DIR}/images"
OUTPUT_DIR="${BASE_DIR}/experiments"
declare -a video_list_files=(
    "experiments/train.txt"
    "experiments/test.txt")

# create train and test dir with symlinks
for video_list_file in "${video_list_files[@]}"
do
    video_list_file_path="${BASE_DIR}/${video_list_file}"
    basename=${video_list_file_path##*/}
    basename=${basename%.*}
    output_base_dir="${OUTPUT_DIR}/${basename}"
    echo "processing $video_list_file_path. creating dir $output_base_dir"
    mkdir ${output_base_dir}
    while read video_name; do
        echo "${output_base_dir}/${video_name} -> ${IMAGE_DIR}/${video_name}"
        ln -s ${IMAGE_DIR}/${video_name} ${output_base_dir}/${video_name}
    done < ${video_list_file_path}
done
