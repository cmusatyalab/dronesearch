INPUT_DIR="/home/junjuew/mobisys18/dataset/3K_VehicleDetection_dataset/train_rgb_annotations_separated"
OUTPUT_DIR="/home/junjuew/mobisys18/dataset/3K_VehicleDetection_dataset/train_rgb_annotations"


declare -a arr=("4K0G0010" 
"4K0G0020" 
"4K0G0030" 
"4K0G0040" 
"4K0G0051" 
"4K0G0060" 
"4K0G0070" 
"4K0G0080" 
"4K0G0090" 
"4K0G0100")

## now loop through the above array
for prefix in "${arr[@]}"
do
   echo "${prefix}"
   cat ${INPUT_DIR}/*${prefix}* | grep "^[0-9]" > ${OUTPUT_DIR}/${prefix}.txt
done
