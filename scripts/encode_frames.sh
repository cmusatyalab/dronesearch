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
base_dir=$1
output_dir=$2

for image_path in $base_dir/*/*;
do
    image_id=$(basename $image_path)
    image_id="${image_id%.*}"    
    dataset_path=$(dirname $image_path)
    dataset_id=$(basename $dataset_path)
    echo ${dataset_id}
    echo ${image_id}
    mkdir -p ${output_dir}/${dataset_id}
    ffmpeg -i ${image_path} -vcodec libx264 -threads 0 -y ${output_dir}/${dataset_id}/${image_id}.mp4 > /dev/null 2>&1
done
