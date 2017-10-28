ANNOTATION_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/annotations_car_only/nexus"
for i in {0..11}; do wc -l $ANNOTATION_DIR/video$i/annotations.txt >> car_annotations_stats.txt; done
