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
video_ids=(
    "little_video1"
    "nexus_video1"
    "hyang_video5"
    "little_video3"
    "gates_video5"
    "gates_video6"
    "gates_video0"
    "gates_video3"
    "coupa_video2"
    "coupa_video1"
    "deathCircle_video0"
    "bookstore_video2"
    "bookstore_video3"
    "bookstore_video0"
    "bookstore_video1"
    "deathCircle_video3"
    "little_video2"
    "hyang_video4"
    "bookstore_video5"
    "bookstore_video4"
    "gates_video1"
)
for video_id in "${video_ids[@]}"
do
    echo ${video_id}
    python preprocess.py resize-stanford-frame-sequence-by-id stanford/images stanford/images_448_224 $video_id 448 224 &
done
wait
