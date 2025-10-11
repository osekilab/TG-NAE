#!/bin/bash

set verbose
set -o errexit

source ./src/syntactic_attention_based_metric_transformer_grammars/.tgenv/bin/activate

python ./src/syntactic_attention_based_metric_transformer_grammars/train.py --config ./experiments/tg/seed=0/config_tg.py
