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

## Tile-based experiments on Stanford with Mobilenet

   ```
   # crop full resolution train and test images into smaller tiles
   # after cropping, some tiles in full resolution positive examples now are negatives as well
   OLDIFS=$IFS; IFS=',';
   for i in train,positive train,negative test,positive test,negative; do   
      set -- $i;
      stage=$1
      category=$2
      echo "slicing full resolution images from $stage $category"
      full_resolution_dir=/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/mobilenet_classification/${stage}/photos/${category}
      output_tile_dir=/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_${stage}_${category}_sliced
      python preprocess.py slice-images ${full_resolution_dir} ${output_tile_dir} --slice-w=0.25 --slice-h=0.25 --slice_is_ratio=True
   done
   IFS=$OLDIFS

   # group tile positive examples together from all tile images from positive full res images
   # for train
   tiled_stanford=/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification
   python preprocess.py group-sliced-images-by-label stanford ${tiled_stanford}/full_resolution_train_positive_sliced /home/junjuew/mobisys18/processed_dataset/stanford_campus/annotations ${tiled_stanford}sliced_train_positive
   # for test
   python preprocess.py group-sliced-images-by-label stanford ${tiled_stanford}/full_resolution_test_positive_sliced /home/junjuew/mobisys18/processed_dataset/stanford_campus/annotations ${tiled_stanford}/sliced_test_positive

   # merge tiles in full resolution positive examples that don't actually have object of interests into negative set
   cd /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/
   mkdir train_negative
   cd train_negative
   find /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_train_negative_sliced/ -type f -exec ln -s '{}' . \;
   find /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/1_discard_small_overlap_roi/full_resolution_train_positive_regrouped_by_tiles/negative/ -type l -exec ln -s '{}' . \;

   mkdir test_negative
   cd test_negative
   find /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_test_negative_sliced/ -type f -exec ln -s '{}' . \;
   find /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/1_discard_small_overlap_roi/full_resolution_test_positive_regrouped_by_tiles/negative/ -type l -exec ln -s '{}' . \;
   
   # sample positive and negative examples
   cd ~/mobisys/scripts
   tiled_stanford=/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/1_discard_small_overlap_roi
   python preprocess.py sample-files-from-directory $tiled_stanford/train_negative $tiled_stanford/train/photos/negative 20000
   python preprocess.py sample-files-from-directory $tiled_stanford/train_positive $tiled_stanford/train/photos/positive 10000

   # To train mobilenet, go to tf-slim dir first
   TFSLIM="${MOBISYS}/scripts/mobilenet/research/slim"
   cd $TFSLIM

   # convert data format into tf-record for tensorflow training
   python -m datasets/convert_twoclass run --dataset_dir=$tiled_stanford/train --mode=train --validation_percentage=0.1
   python -m datasets/convert_twoclass run --dataset_dir=$tiled_stanford/test --mode=test

   # start finetuning mobilenet
   bash scripts/finetune_mobilenet_v1_on_stanford.sh
   ```

# Freeze and optimize trained model
python freeze_and_optimize_deploy_model.py --checkpoint_path $tiled/2_more_test/mobilenet_train/logs_last_layer_only_150k --dataset_dir $tiled/2_more_test/mobilenet_train --output_dir /tmp/

# Extra resource
  * Tensorflow retrain last layer only: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py

# Things to fix:
  * The preprocessing fn for the mobilenet does a central_crop to crop 0.875 centre images. This effectively remove the training images in which a car just appeared.

# Design of the adaptive pipeline
## Class interfaces
  * InputSource(): input sources for the pipeline
    * __init__(feed_path): open feed
    * read(): get next frame, blocking
    * close(): close feed
  * VideoInputSource(InputSource)
  * FileInputSource(InputSource)
  * Filter(): filter abstraction
    * process(image) -> FilterOutput
    * update(data)
    * close() -> close filter
  * FilterFactory: Create filters based on configuration files on disk
    * create_filter(filter_name)
  * FilterOutput: Output of Filter
    * tobytes() -> serialized bytes
    * frombytes(input_bytes) -> deserialize bytes
  * TileImageFilterOutput: Tile image as output filter
  * NetworkService: Networking to backend
    * __init__() -> get context
    * send(data)
    * recv(data)
  * NetworkFactory: Factory to create networking component
  * CommandQueue: Queue to receive command
    * get()

## Requirements
  * logzero
  * fire
  * opencv 2.4
  * tensorflow
