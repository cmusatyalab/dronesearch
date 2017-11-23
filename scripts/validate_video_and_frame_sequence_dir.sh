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
