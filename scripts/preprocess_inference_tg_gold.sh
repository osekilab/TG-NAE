#!/bin/bash

source .venv/bin/activate

# 1. Extract the gold trees from the natural stories
python src/extract_gold_trees_from_naturalstories.py \
    --input_path ./src/naturalstories/parses/penn/all-parses.txt.penn \
    --output_path ./data/naturalstories/texts/gold_trees.txt

# 2. Extract the id/word mapping from the natural stories
mkdir -p ./data/naturalstories/ids
python src/extract_id_word_pairs_from_naturalstories.py \
    --input_file ./src/naturalstories/parses/penn/all-parses-aligned.txt.penn \
    --output_file ./data/naturalstories/ids/word_ids.csv

# 3. Convert gold parses to TG format
source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

export SPM=./data/bllip-lg/spm

bash ./src/syntactic_attention_based_metric_transformer_grammars/convert_gold_trees2tg_trees.sh \
    --input_file ./data/naturalstories/texts/gold_trees.txt \
    --brackets_dir ./data/naturalstories/texts/parsed_naturalstories/brackets \
    --charniak_dir ./data/naturalstories/texts/parsed_naturalstories/brackets.charniak \
    --output_dir ./data/naturalstories/texts/parsed_naturalstories/charniak \
    --tokenized_dir ./data/naturalstories/texts/parsed_naturalstories/enc \
    --csv_dir ./data/naturalstories/texts/parsed_naturalstories/csv
