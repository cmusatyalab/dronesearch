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
