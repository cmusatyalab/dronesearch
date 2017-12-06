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
