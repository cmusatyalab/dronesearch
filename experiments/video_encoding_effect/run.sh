#!/usr/bin/env bash

# encode the videos into different crf
original_dir="/home/junjuew/mobisys18.old/processed_dataset/okutama/test_videos"
output_dir="/home/junjuew/mobisys18.old/processed_dataset/okutama/crf_videos"
image_output_dir="/home/junjuew/mobisys18.old/processed_dataset/okutama/crf_images"
test_videos=(
"2.2.2"
"2.2.4"
"1.1.7"
)

# The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible. A
# lower value generally leads to higher quality, and a subjectively sane range is 17–28.
# Consider 17 or 18 to be visually lossless or nearly so;
# it should look the same or nearly the same as the input but it isn't technically lossless.
# The range is exponential, so increasing the CRF value +6 results in roughly half the bitrate / file size, while -6
# leads to roughly twice the bitrate.
# https://trac.ffmpeg.org/wiki/Encode/H.264


crfs=(
#"17"
#"23"
#"29"
"35"
#"41"
#"47"
)

function transcode {
    mkdir -p ${output_dir}

    for crf in ${crfs[@]}; do
        mkdir -p ${output_dir}/${crf}_test
        for test_video in ${test_videos[@]};do
            # vsync 0 is needed to make the # of frames the same
            ffmpeg -i ${original_dir}/${test_video}.mp4 -vcodec libx264 -vsync 0 -crf ${crf} \
            ${output_dir}/${crf}_test/${test_video}.mp4 &
        done
    done
}

function extract {
    for crf in ${crfs[@]}; do
        for test_video in ${test_videos[@]};do
            mkdir -p ${image_output_dir}/${crf}/${test_video}
            ffmpeg -i ${output_dir}/${crf}_test/${test_video}.mp4 -qscale:v 2 \
            ${image_output_dir}/${crf}/${test_video}/%10d.jpg &
        done
    done
}

function inference {
declare -A work
work=(
["26"]="17"
["27"]="23"
["28"]="29"
["29"]="35"
["30"]="41"
)
mid="$(hostname | tail -c 3)"
echo {$mid}
work_crf=${work[${mid}]}
echo "working on crf ${work_crf}"
mkdir -p results/crf/${work_crf}
python okutama_inference.py infer --frozen-graph-path \
/home/junjuew/mobisys18/trained_models/frozen_okutama_model/frozen_inference_graph.pb --label-file-path \
/home/junjuew/mobisys18/trained_models/frozen_okutama_model/person_label_map.pbtxt \
--num-classes 1 --image-base-dir \
/home/junjuew/mobisys18/processed_dataset/okutama/crf_images/${work_crf} \
--output-dir results/crf/${work_crf}
}

#work_crf="47"
#echo "working on crf ${work_crf}"
#mkdir -p results/crf/${work_crf}
#python okutama_inference.py infer --frozen-graph-path \
#/home/junjuew/mobisys18/trained_models/frozen_okutama_model/frozen_inference_graph.pb --label-file-path \
#/home/junjuew/mobisys18/trained_models/frozen_okutama_model/person_label_map.pbtxt \
#--num-classes 1 --image-base-dir \
#/home/junjuew/mobisys18/processed_dataset/okutama/crf_images/${work_crf} \
#--output-dir results/crf/${work_crf}

inference
