# train arguments
# BASE_DIR="/home/junjuew/mobisys18/processed_dataset/munich"
# INPUT_DIR="${BASE_DIR}/train_rgb_annotations_separated"
# OUTPUT_DIR="${BASE_DIR}/train_rgb_annotations"
# declare -a train_set=("4K0G0010" 
# "4K0G0020" 
# "4K0G0030" 
# "4K0G0040" 
# "4K0G0051" 
# "4K0G0060" 
# "4K0G0070" 
# "4K0G0080" 
# "4K0G0090" 
# "4K0G0100")


# test arguments
BASE_DIR="/home/junjuew/mobisys18/processed_dataset/munich"
INPUT_DIR="${BASE_DIR}/test_rgb_annotations_separated"
OUTPUT_DIR="${BASE_DIR}/test_rgb_annotations"
declare -a test_set=(
"4K0G0110" 
"4K0G0120" 
"4K0G0130" 
"4K0G0140" 
"4K0G0150" 
"4K0G0160" 
"4K0G0250" 
"4K0G0265" 
"4K0G0278" 
"4K0G0285")

## now loop through the above array
for prefix in "${test_set[@]}"
do
   echo "${prefix}"
   cat ${INPUT_DIR}/*${prefix}* | grep "^[0-9]" > ${OUTPUT_DIR}/${prefix}.txt
done
