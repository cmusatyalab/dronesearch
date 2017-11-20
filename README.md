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

   # prepare JIT training images. Pick one image ID from the original test set.
   scripts/munich_make_jit_data.sh 4K0G0250

   # To train mobilenet, go to tf-slim dir first
   TFSLIM="${MOBISYS}/scripts/mobilenet/research/slim/scripts"
   cd $TFSLIM

   # convert data format into tf-record for tensorflow training
   python download_and_convert_data.py --dataset_name=munich --dataset_dir=$PROCESSED/munich/mobilenet_train

   # start finetuning mobilenet
   bash scripts/finetune_mobilenet_v1_on_munich.sh
   
   # Start evaluation loop (train, validation, test) in another terminal
   
   
   # start JIT training mobilenet
   scripts/jit_train_mobilenet_v1_on_munich.sh 4K0G0120
   ```

## SVM predicting TP and FP from DNN using Mobilenet's 1024-D pre-logits

    ```bash
    REDIS_HOST=172.17.0.10
    TEST_VIDEO=bookstore_video2
    
    # Prepare SVM training data in a pickle file.
    python prepare_jit_train_data.py make_tp_fp_dataset \
        --redis-host ${REDIS_HOST} \
        --over_sample_ratio=20 \
        --file_glob "/home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/tile_test_by_label/*/${TEST_VIDEO}*" \
        --output_file /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/${TEST_VIDEO}_tp_fp.p
    
    # Train/Eval SVM on the dataset generated from the above command
    python svm_on_pre_logit.py train \
        --pre_logit_files /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/${TEST_VIDEO}_tp_fp.p \
        --test_ratio 0.3 \
        --eval_every_iters=10
    
    # The above two commands are wrapped in scripts/jit_svm_on_stanford.sh
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
   python preprocess.py group-sliced-images-by-label stanford /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_train_positive_sliced /home/junjuew/mobisys18/processed_dataset/stanford_campus/annotations /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/sliced_train_positive
   # for test
   python preprocess.py group-sliced-images-by-label stanford /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_test_positive_sliced /home/junjuew/mobisys18/processed_dataset/stanford_campus/annotations /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/sliced_test_positive

   # merge tiles in full resolution positive examples that don't actually have object of interests into negative set
   cd /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/
   mkdir train_negative
   find sliced_train_positive/negative/ -name '*.*' | xargs mv -t train_negative/
   find full_resolution_train_negative_sliced/ -name '*.*' | xargs mv -t train_negative/
   mv sliced_train_positive/positive train_positive
   rm -rf sliced_train_positive

   mv full_resolution_test_negative_sliced test_negative
   find sliced_test_positive/negative/ -name '*.*' | xargs mv -t test_negative/
   mv sliced_test_positive/positive test_positive
   rm -rf sliced_test_positive

   # sample positive and negative examples
   cd ~/mobisys/scripts
   python preprocess.py sample-files-from-directory /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train_negative /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train/photos/negative 20000
   python preprocess.py sample-files-from-directory /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train_positive /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train/photos/positive 10000
   ```


# Extra resource
  * Tensorflow retrain last layer only: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py



python preprocess.py sample-files-from-directory /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/full_resolution_train_positive_sliced /home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train_positive_samples_10000 10000
