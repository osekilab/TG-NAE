#!/bin/bash

source .venv/bin/activate

# 1. Extract the terminals from the natural stories
python ./src/extract_terminals_from_naturalstories.py \
    --natural_stories_ptb_path ./src/naturalstories/parses/penn/all-parses.txt.penn \
    --output_path ./data/naturalstories/texts/terminals.txt

set verbose
set -o errexit

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

# Tokenize the data from Choe-Charniak format to space-separated integers.
spm_encode \
  --output_format=id \
  --model=./data/bllip-lg/spm/spm.model \
  --input=./data/naturalstories/texts/terminals.txt \
  --output=./data/naturalstories/texts/terminals.enc

# Remove the redundant whitespace, output as CSV.
python ./src/syntactic_attention_based_metric_transformer_grammars/tools/postprocess_encoded_docs.py \
  --input ./data/naturalstories/texts/terminals.enc \
  --output ./data/naturalstories/texts/terminals.csv \
  --vocab ./data/bllip-lg/spm/spm.vocab
