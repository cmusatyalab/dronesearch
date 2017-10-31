# Overview

These are the experiments and scripts for mobisys'18

# What's in the directory
  * dataset: downloaded dataset without any modifications
  * processed_dataset: preprocessed datasets for experiments and experiments results
  * scripts: scripts used to do preprocess and launch experiments
  * environment: environment and packages for running the experiment. tf1.3-gpu-env is the virtualenv for tensorflow
  * pretrained_models: downloaded pretrained models without any modifications
  * py-faster-rcnn: Object Detection py-faster-rcnn code. TODO: move to scripts directory
  
# How to Run Experiment

## Classification on Munich Dataset
   Please double-check/change the output directory before running these commands. They may accidentally overwrite existing experiment data.
   ```
   MOBISYS="/home/junjuew/mobisys18"
   PROCESSED="${MOBISYS}/processed_dataset"

   cd scripts

   # crop images into smaller blocks
   python preprocess.py slice-images ../processed_dataset/munich/train_rgb ../processed_dataset/munich/train_sliced

   # combine original annotation into one big file
   bash munich_combine_annotations.sh

   # visualize ground truth annotations
   python visualize.py visualize-annotations-in-image $PROCESSED/munich/train_rgb/2012-04-26-Muenchen-Tunnel_4K0G0010.JPG $PROCESSED/munich/train_rgb_annotations/4K0G0010.txt munich $PROCESSED/munich/annotated_4K0G0010.jpg

   # prepare images for training. put positive and negative classification examples into different dir
   python preprocess.py group-sliced-images-by-label $PROCESSED/munich/train_sliced $PROCESSED/munich/train_rgb_annotations $PROCESSED/munich/mobilenet_train/photos

   # To train mobilenet, go to tf-slim dir first
   TFSLIM="${MOBISYS}/scripts/mobilenet/research/slim/scripts"
   cd $TFSLIM

   # convert data format into tf-record for tensorflow training
   python download_and_convert_data.py --dataset_name=munich --dataset_dir=$PROCESSED/munich/mobilenet_train

   # start finetuning mobilenet
   bash scripts/finetune_mobilenet_v1_on_munich.sh
   ```

# Extra resource
  * Tensorflow retrain last layer only: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

