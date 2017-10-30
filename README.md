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
   cd scripts

   # crop images into smaller blocks
   python preprocess.py slice-images /home/junjuew/mobisys18/dataset/3K_VehicleDetection_dataset/train_rgb ../processed_dataset/munich/sliced

   # combine original annotation into one big file
   bash munich_combine_annotations.sh

   # visualize ground truth annotations
   python visualize.py visualize-annotations-in-image /home/junjuew/mobisys18/dataset/3K_VehicleDetection_dataset/train_rgb/2012-04-26-Muenchen-Tunnel_4K0G0010.JPG /home/junjuew/mobisys18/dataset/3K_VehicleDetection_dataset/train_rgb_annotations/4K0G0010.txt munich /home/junjuew/mobisys18/dataset/../processed_dataset/munich/annotated_4K0G0010.jpg

   # prepare images for training. put positive and negative classification examples into different dir
   python preprocess.py group-sliced-images-by-label ../processed_dataset/munich/sliced ../processed_dataset/munich/annotations ../processed_dataset/munich/mobilenet_train

   # To train mobilenet, go to tf-slim dir first
   cd scripts/mobilenet/research/slim/scripts

   # convert data format into tf-record for tensorflow training
   python download_and_convert_data.py --dataset_name=munich --dataset_dir=~/mobisys18/processed_dataset/munich/mobilenet_train

   # start finetuning mobilenet
   cd ..
   bash scripts/finetune_mobilenet_v1_on_munich.sh
   ```

# Extra resource
  * Tensorflow retrain last layer only: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

