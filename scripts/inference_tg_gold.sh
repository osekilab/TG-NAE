#!/bin/bash

mkdir -p ./outputs/seed=42/tg_scores

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

bash ./src/syntactic_attention_based_metric_transformer_grammars/run_scoring.sh \
    --csv_dir ./data/naturalstories/texts/parsed_naturalstories/csv \
    --score_output_dir ./outputs/seed=42/tg_scores \
    --checkpoint ./experiments/tg/seed=42/checkpoint_tg.pkl \
    --tokenizer ./data/bllip-lg/spm/spm.model
