#!/bin/bash

set verbose
set -o errexit

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

python ./src/syntactic_attention_based_metric_transformer_grammars/train.py --config ./experiments/txl_tree/seed=42/config_txl_tree.py
