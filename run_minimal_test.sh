#!/bin/bash

PRETRAINED_CKPT="ckpt/pretrained.pth"
TRAIN_JSON="data/coswara_mini_experiment.json"
EVAL_JSON="data/coswara_mini_experiment.json"
LABEL_CSV="data/covid_labels.csv"

OUTPUT_DIR="output/minimal_test_$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR=$OUTPUT_DIR

export OMP_NUM_THREADS=1

uv run main_finetune_as.py \
--log_dir $LOG_DIR \
    --output_dir $OUTPUT_DIR \
    --device 'cpu' \
    --num_workers 2 \
    --model vit_base_patch16 \
    --finetune $PRETRAINED_CKPT \
    --data_train $TRAIN_JSON \
    --data_eval $EVAL_JSON \
    --label_csv $LABEL_CSV \
    --nb_classes 2 \
    --batch_size 4 \
    --epochs 3 \
    --blr 1e-3 \
    --warmup_epochs 1 \
    --first_eval_ep 1 \
    --mask_2d True \
    --roll_mag_aug False \
    --dataset audioset
