#!/bin/bash

DATASET_NAME="coughvid" 
PRETRAINED_CKPT="ckpt/pretrained.pth"
LABEL_CSV="data/covid_labels.csv"
BALANCED=False
if [ "$DATASET_NAME" == "coswara" ]; then
    if [ "$BALANCED" == True ]; then
        echo "Treinando com balanceamento - COSWARA"
        TRAIN_JSON="data/coswara_train.json"
        EVAL_JSON="data/coswara_eval.json"
        OUTPUT_SUFFIX="coswara"
    else
        echo "Treinando sem balanceamento - COSWARA"
        TRAIN_JSON="data/coswara_train_no_balance.json"
        EVAL_JSON="data/coswara_eval_no_balance.json"
        OUTPUT_SUFFIX="coswara_no_balance"
    fi
elif [ "$DATASET_NAME" == "coughvid" ]; then
    if [ "$BALANCED" == True ]; then
        echo "Treinando com balanceamento - COUGHVID"
        TRAIN_JSON="data/coughvid_train.json"
        EVAL_JSON="data/coughvid_eval.json"
        OUTPUT_SUFFIX="coughvid"
    else
        echo "Treinando sem balanceamento - COUGHVID"
        TRAIN_JSON="data/coughvid_train_no_balance.json"
        EVAL_JSON="data/coughvid_eval_no_balance.json"
        OUTPUT_SUFFIX="coughvid_no_balance"
    fi
else
    echo "Erro: DATASET_NAME inválido. Escolha 'coswara' ou 'uk_covid'."
    exit 1
fi

RUN_NAME="${OUTPUT_SUFFIX}_finetune_$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="output/${RUN_NAME}"
LOG_DIR=$OUTPUT_DIR

BATCH_SIZE=32       # Aumente conforme a memória da sua GPU permitir (ex: 32, 64)
EPOCHS=60           # Número de epochs (baseado no ft_as.sh)
BASE_LR=2e-4        # Taxa de aprendizagem base (menor, bom para finetuning) era 2e-4
WARMUP_EPOCHS=4     # Epochs de aquecimento (baseado no ft_as.sh)
FIRST_EVAL_EP=15    # Começar a avaliar após esta epoch (baseado no ft_as.sh)
NUM_WORKERS=4       # Número de workers para o DataLoader (ajuste conforme seu CPU/memória)

export OMP_NUM_THREADS=1

uv run main_finetune_as.py \
    --log_dir $LOG_DIR \
    --output_dir $OUTPUT_DIR \
    --device 'cuda' \
    --model vit_base_patch16 \
    --finetune $PRETRAINED_CKPT \
    --data_train $TRAIN_JSON \
    --data_eval $EVAL_JSON \
    --label_csv $LABEL_CSV \
    --nb_classes 2 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --blr $BASE_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --first_eval_ep $FIRST_EVAL_EP \
    --num_workers $NUM_WORKERS \
    --mixup 0.5 \
    --mask_2d True \
    --roll_mag_aug False \
    --dataset audioset \
    --pin_mem

echo "Treino concluído. Resultados em: $OUTPUT_DIR"