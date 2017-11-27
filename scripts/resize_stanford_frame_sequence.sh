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
