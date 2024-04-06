## Data processing

1、Multimodal features are extracted using a pre-trained model that performs the:

```bash
cd src
python img_encoder.py
python text_encoder.py
```

Note that you will need to download the pre-trained model([CLIP](https://github.com/openai/CLIP)) and dataset as well as change your catalogue。

You can get FB15K-237-IMG from [MKGformer](https://github.com/zjunlp/MKGformer) ,get WN18-IMG from [RSME](https://github.com/wangmengsd/RSME) And you can get WN9-IMG from [IKRL](https://github.com/thunlp/IKRL)



2、You can complete the preprocessing by executing the following command

```bash
python process_datasets.py
```



## Train

```bash
CUDA_LAUNCH_BLOCKING=1 python learn.py --model RAGCA --ckpt_dir ../checkpoint/ --dataset FB15K-237 --early_stopping 20 --reg=0.08

CUDA_LAUNCH_BLOCKING=1 python learn.py --model RAGCA --ckpt_dir ../checkpoint/ --dataset WN18 --early_stopping 20 --img_info ../embedings/WN18/CLIP_img_feature.pickle --dscp_info ../embedings/WN18/CLIP_description_feature.pickle --node_info ../embedings/WN18/CLIP_entity_text_feature.pickle --rel_desc_info ../embedings/WN18/CLIP_relation_text_feature.pickle  --reg 0.08

CUDA_LAUNCH_BLOCKING=1 python learn.py --model RAGCA --ckpt_dir ../checkpoint/ --dataset WN9 --early_stopping 20 --img_info ../embedings/WN9/CLIP_img_feature.pickle --dscp_info ../embedings/WN9/CLIP_description_feature.pickle --node_info ../embedings/WN9/CLIP_entity_text_feature.pickle --rel_desc_info ../embedings/WN9/CLIP_relation_text_feature.pickle  --reg 0.08

```



