* test_classification_speed.py: is used for inference_speed using tensorflow evaluated on mobilenet and incpetion.
* test_object_detection_speed.py: is used for inference_speed using tensorflow evaluated on SSD + mobilenet and SSD + incpetion, Faster-RCNN + inception and Faster-RCNN + ResNet101.

Sample images can be found at junjuew@cloudlet001.elijah.cs.cmu.edu:/home/junjuew/sample-images
Pretrained models can be found at junjuew@cloudlet001.elijah.cs.cmu.edu:/home/junjuew/mobisys18/pretrained_models

Sample commands:
* Image classification:
```
python test_classification_speed.py \
--graph=/home/junjuew/mobisys18/pretrained_models/inception_v2_2016_08_28_frozen.pb 2>&1 \
| tee ~/mobisys18/experiments/inference_speed/tf_inceptionv2_k40.txt
```
```
python test_classification_speed.py \
--model-file=/home/junjuew/mobisys18/pretrained_models/frozen_mobilenet_v1.pb \
--input_mean=128 --input_std=128 --output-layer=MobilenetV1/Predictions/Reshape_1 2>&1 \
| tee ~/mobisys18/experiments/inference_speed/tf_mobilenetv1_.txt
```
```bash
python test_classification_speed.py \
--model-file=/home/junjuew/mobisys18/pretrained_models/frozen_resnet_v1_101.pb \
--output-layer=resnet_v1_101/predictions/Reshape_1 2>&1 \
| tee ~/mobisys18/experiments/inference_speed/tf_resnet_v1_101_.txt
```
* Object Detection:
```
python test_object_detection_speed.py 2>&1 | tee ~/mobisys18/experiments/inference_speed/tf_faster_rcnn_resnet101_k40.txt
```


## Output Layer Node Names
* MobileNet: MobilenetV1/Predictions/Reshape_1
* Resnet 101: resnet_v1_101/predictions/Reshape_1
* Inception V3: InceptionV3/Predictions/Reshape_1
