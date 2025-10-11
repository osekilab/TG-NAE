#!/bin/sh
# Copyright 2021-2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


set verbose
set -o errexit

DATA=./data/naturalstories/texts
TOKENIZER=spm

mkdir -p ./outputs/seed=42/txl_scores

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

python ./src/syntactic_attention_based_metric_transformer_grammars/score.py \
    --checkpoint ./experiments/txl/seed=42/checkpoint_txl.pkl \
    --tokenizer ./data/bllip-lg/spm/spm.model \
    --input $DATA/terminals.csv \
    --output ./outputs/seed=42/txl_scores/
