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
    BASE_DIR="/home/junjuew/mobisys18/processed_dataset/okutama"
fi
VIDEO_DIR="${BASE_DIR}/videos/"

# validate # of frames in the video and extract image dir
total_frame_num=0
for video_path in ${VIDEO_DIR}*
do
    video_num_frames="$(ffprobe -select_streams v -show_streams ${video_path} 2>/dev/null | grep nb_frames | sed -e 's/nb_frames=//')"
    echo "${video_path#${VIDEO_DIR}}: ${video_num_frames}"
    let total_frame_num+=${video_num_frames}
done
echo "total video frames: ${total_frame_num}"
