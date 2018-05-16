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

## Android
Push the modified file in [android](android) into tensorflow Android example app.
The experiment version used tf 1.8.0 release.
https://github.com/tensorflow/tensorflow/tree/r1.8/tensorflow/examples/android

## Working Git Repos for Tensorflow Wheel for Jetson TX2
* 1.8: https://github.com/peterlee0127/tensorflow-nvJetson
    * It takes a long time when creating a tf session. patience is needed
* 1.7: https://devtalk.nvidia.com/default/topic/1031300/tensorflow-1-7-wheel-with-jetpack-3-2-/
* 1.6: https://github.com/peterlee0127/tensorflow-nvJetson
* 1.5: https://github.com/Davidnet/JetsonTFBuilds/tree/master/official
* Not working wheels. The installed packages hang when creating a tf session:
    * All wheels from https://github.com/peterlee0127/tensorflow-nvJetson (except 1.6 and 1.8)
    * https://github.com/openzeka/Tensorflow-for-Jetson-TX2

## Notes On Jetson
1. Jetson has multiple power model. Need to use nvpmodel to adjust to make sure the highest performance (MAXN). See http://www.jetsonhacks.com/2017/03/25/nvpmodel-nvidia-jetson-tx2-development-kit/.
2. Tensorflow needs to use gpu_options.allow_growth. Otherwise, it would have difficulties allocating enough CUDA memories.
```python
    # needed for jetson to be able to allocate enough memory
    # see https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=detection_graph, config=config) as sess:
        ...
```
3. Remember to use jetson_clocks.sh to start the fan when choosing the highest performance model.

## Notes on Server
1. Set the performance to be highest use:
```bash
sudo nvidia-smi -ac 3004,875
```


