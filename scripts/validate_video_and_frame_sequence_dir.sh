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
if [[ -z "$BASE_DIR" ]]; then
    BASE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus"
fi
if [[ -z "$VIDEO_LIST_FILE" ]]; then
    $VIDEO_LIST_FILE="${BASE_DIR}/video_with_cars.txt"
fi
VIDEO_DIR="${BASE_DIR}/videos"
OUTPUT_DIR="${BASE_DIR}/images"

# validate # of frames in the video and extract image dir
total_video_num=0
while read video_name; do
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/${video_name}"
    echo "${VIDEO_DIR}/${video_name} v.s. ${VIDEO_OUTPUT_DIR}"
    video_num_frames="$(ffprobe -select_streams v -show_streams ${VIDEO_DIR}/${video_name} 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//')"
    image_dir_num_frames="$(ls -A $VIDEO_OUTPUT_DIR | wc -l)"
    echo "video_num_frames: ${video_num_frames}, image_dir_num_frames: ${image_dir_num_frames}"
    if [[ "${video_num_frames}" -ne "${image_dir_num_frames}" ]]; then
        echo "${video_name} convertion wrong!"
    fi
    let total_video_num+=${video_num_frames}
done < ${VIDEO_LIST_FILE}
echo "total video frames ${total_video_num}"
wait
