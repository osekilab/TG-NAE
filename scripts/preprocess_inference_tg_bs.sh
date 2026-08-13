#!/bin/bash

set -e  # Set script to stop immediately on error

RNNG=./src/syntactic_attention_based_metric_rnng-pytorch
TG=./src/syntactic_attention_based_metric_transformer_grammars
PARSES=./outputs/parsed_naturalstories_bs

source .venv/bin/activate

# 1. Extract the terminals from the natural stories
mkdir -p ./data/naturalstories/texts
python ./src/extract_terminals_from_naturalstories.py \
    --natural_stories_ptb_path ./src/naturalstories/parses/penn/all-parses.txt.penn \
    --output_path ./data/naturalstories/texts/terminals.txt

# 2. Extract the id/word mapping from the natural stories
mkdir -p ./data/naturalstories/ids
python ./src/extract_id_word_pairs_from_naturalstories.py \
    --input_file ./src/naturalstories/parses/penn/all-parses-aligned.txt.penn \
    --output_file ./data/naturalstories/ids/word_ids.csv

# 3. Parse the natural stories with word-synchronous beam search
#
# Besides the surprisals and the single best parse per sentence, this writes
# surprisals_naturalstories.txt.marginalized, holding the parses still active
# in the beam at each word. Those are what the NAE is averaged over.
source .rnngenv/bin/activate

mkdir -p ${PARSES}
python ${RNNG}/beam_search.py \
    --test_file ./data/naturalstories/texts/terminals.txt \
    --lm_output_file ${PARSES}/surprisals_naturalstories.txt \
    --model_file ./experiments/rnng/rnng.pt \
    --beam_size 100 \
    --word_beam_size 10 \
    --shift_size 1 \
    --batch_size 20 \
    --block_size 500 \
    --gpu 0 \
    --seed 42 > ${PARSES}/final_top_parse_naturalstories.txt

# 4. Convert the RNNG parses to TG format
source ${TG}/.tgenv/bin/activate

export SPM=./data/bllip-lg/spm

bash ${TG}/convert_rnng_trees2tg_trees.sh \
    --input_file ${PARSES}/surprisals_naturalstories.txt.marginalized \
    --brackets_dir ${PARSES}/brackets \
    --charniak_dir ${PARSES}/brackets.charniak \
    --output_dir ${PARSES}/charniak \
    --tokenized_dir ${PARSES}/enc \
    --csv_dir ${PARSES}/csv
