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
