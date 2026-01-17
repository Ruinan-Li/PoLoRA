#!/bin/bash
export LANG=zh_CN.UTF-8

python main.py -model_name LoraKGE -ent_r 40 -rel_r 40 -gpu 1 -dataset HYBRID -lora_lr 1.5
python main.py -model_name LoraKGE -ent_r 40 -rel_r 40 -gpu 0 -dataset WN_CKGE -lora_lr 1.5