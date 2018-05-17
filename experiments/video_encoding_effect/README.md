# Freeze the trained model
```bash
python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path /home/junjuew/mobisys18/trained_models/okutama_model/faster_rcnn_resnet101_voc07.config \
--trained_checkpoint_prefix /home/junjuew/mobisys18/trained_models/okutama_model/model.ckpt-71132 \
--output_directory /home/junjuew/mobisys18/trained_models/okutama_model/frozen_okutama_model
```
