if [[ -z "$BASE_DIR" ]]; then
    BASE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus"
fi
if [[ -z "$VIDEO_LIST_FILE" ]]; then
    $VIDEO_LIST_FILE="${BASE_DIR}/video_with_cars.txt"
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
