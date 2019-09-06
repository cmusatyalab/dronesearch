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
    VIDEO_LIST_FILE="${BASE_DIR}/video_with_cars.txt"
fi

VIDEO_DIR="${BASE_DIR}/videos"
OUTPUT_DIR="${BASE_DIR}/images"

# extract video into images
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir ${OUTPUT_DIR}
fi

while read video_name; do
    VIDEO_OUTPUT_DIR="${OUTPUT_DIR}/${video_name}"
    if [[ ! -d "$VIDEO_OUTPUT_DIR" ]]; then
        mkdir ${VIDEO_OUTPUT_DIR}
    fi
    echo "${VIDEO_DIR}/${video_name} --> ${VIDEO_OUTPUT_DIR}"
    avconv -i ${VIDEO_DIR}/${video_name} -f image2 ${VIDEO_OUTPUT_DIR}/%10d.jpg &
done < ${VIDEO_LIST_FILE}
wait
