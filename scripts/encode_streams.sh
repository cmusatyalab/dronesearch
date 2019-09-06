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

for stream_dir in $base_dir/*/*/*;
do
    stream_id=$(basename $stream_dir)
    video_dir=$(dirname $stream_dir)
    video_id=$(basename $video_dir)
    dataset_dir=$(dirname $video_dir)
    dataset_id=$(basename $dataset_dir)
    echo ${dataset_id}
    echo ${video_id}
    echo ${stream_id}
    mkdir -p ${output_dir}/${dataset_id}
    ffmpeg -r 30 -i ${stream_dir}/%010d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ${output_dir}/${dataset_id}/${video_id}_${stream_id}.mp4 &
done
wait
