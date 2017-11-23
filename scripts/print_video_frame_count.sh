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
