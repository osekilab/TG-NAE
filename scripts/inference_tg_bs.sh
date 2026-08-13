#!/bin/bash

mkdir -p ./outputs/seed=42/tg_scores_bs

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

bash ./src/syntactic_attention_based_metric_transformer_grammars/run_scoring.sh \
    --csv_dir ./outputs/parsed_naturalstories_bs/csv \
    --score_output_dir ./outputs/seed=42/tg_scores_bs \
    --checkpoint ./experiments/tg/seed=42/checkpoint_tg.pkl \
    --tokenizer ./data/bllip-lg/spm/spm.model
