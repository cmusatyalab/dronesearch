# video_ids=(
#     "01"
#     "02"
#     "03"
#     "04"
#     "05"
#     "06"
#     "07"
#     "08"
#     "09"
#     "10"
#     "11"
# )
video_ids=(
    '1.1.3'
    '1.1.2'
    '1.1.5'
    '1.1.4'
    '2.2.7'
    '2.1.7'
    '2.1.4'
    '2.2.11'
    '1.1.10'
    '2.2.2'
    '2.2.4'
    '1.1.7'
)
dataset_dir=$1
long_edge=$2
short_edge=$3
for video_id in "${video_ids[@]}"
do
    echo ${video_id}
    python preprocess.py resize-frame-sequence-by-id ${dataset_dir}/images ${dataset_dir}/images_${long_edge}_${short_edge} "$video_id" ${long_edge} ${short_edge} &
done
wait
