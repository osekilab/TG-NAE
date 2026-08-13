#!/bin/bash

set -e  # Set script to stop immediately on error

# Activate virtual environment
if ! source .rnngenv/bin/activate; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

RNNG=./src/syntactic_attention_based_metric_rnng-pytorch
DATA=./data/bllip-lg/vocabsize=20000.unkmethod=subword.keep_ptb_bracket=True
EXP=./experiments/rnng

mkdir -p ${EXP}

python ${RNNG}/train.py \
    --train_file ${DATA}-train.json \
    --val_file ${DATA}-val.json \
    --sp_model ${DATA}-spm.model \
    --fixed_stack \
    --strategy top_down \
    --w_dim 256 \
    --h_dim 256 \
    --num_layers 2 \
    --dropout 0.1 \
    --composition lstm \
    --batch_group similar_action_length \
    --group_sentence_size 4096 \
    --optimizer adam \
    --batch_size 512 \
    --batch_action_size 26000 \
    --save_path ${EXP}/rnng.pt \
    --num_epochs 40 \
    --lr 0.001 \
    --gpu 0 \
    --device cuda \
    --seed 42 \
    --print_every 500 \
    --tensorboard_log_dir ${EXP}/rnng \
    --early_stop \
    --early_stop_patience 5
