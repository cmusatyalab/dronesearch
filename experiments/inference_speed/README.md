* test_classification_speed.py: is used for inference_speed using tensorflow evaluated on mobilenet and incpetion.
* test_object_detection_speed.py: is used for inference_speed using tensorflow evaluated on SSD + mobilenet and SSD + incpetion, Faster-RCNN + inception and Faster-RCNN + ResNet101.

Sample images can be found at junjuew@cloudlet001.elijah.cs.cmu.edu:/home/junjuew/sample-images
Pretrained models can be found at junjuew@cloudlet001.elijah.cs.cmu.edu:/home/junjuew/mobisys18/pretrained_models

Sample commands:
* Image classification:
```
python test_classification_speed.py --graph=/home/junjuew/mobisys18/pretrained_models/inception_v2_2016_08_28_frozen.pb 2>&1 | tee ~/mobisys18/experiments/inference_speed/tf_inceptionv2_k40.txt
```
```
python test_classification_speed.py --graph=/home/junjuew/mobisys18/pretrained_models/frozen_mobilenet_v1.pb --input_mean=128 --input_std=128 --output_layer=MobilenetV1/Predictions/Reshape_1
```
* Object Detection:
```
python test_object_detection_speed.py 2>&1 | tee ~/mobisys18/experiments/inference_speed/tf_faster_rcnn_resnet101_k40.txt
```


