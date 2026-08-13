#!/bin/bash

set -e  # Set script to stop immediately on error

# Activate virtual environment
if ! source .rnngenv/bin/activate; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

RNNG=./src/syntactic_attention_based_metric_rnng-pytorch
DATA=./data/bllip-lg
OUT=${DATA}/vocabsize=20000.unkmethod=subword.keep_ptb_bracket=True

# Convert the Penn-style trees into the json files the RNNG reads, and train a
# SentencePiece model on the terminals so that rare words are split into
# subwords rather than replaced by a single unknown token.
#
# Writes ${OUT}-train.json, ${OUT}-val.json, ${OUT}-test.json and
# ${OUT}-spm.model, all of which train_rnng.sh reads.
python ${RNNG}/preprocess.py \
    --vocabsize 20000 \
    --unkmethod subword \
    --keep_ptb_bracket \
    --trainfile ${DATA}/train.txt \
    --valfile ${DATA}/valid.txt \
    --testfile ${DATA}/test.txt \
    --outputfile ${OUT}
