# Computing NAE with Transformer Grammar

Repository for computing Normalized Attention Entropy (NAE) with Transformer Grammar (TG).

Official implementation for the ACL 2025 paper:
[**"If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?"**](https://aclanthology.org/2025.acl-long.483/). This implementation builds upon [Transformer Grammars: Augmenting Transformer Language Models with Syntactic Inductive Biases at Scale](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00526/114315/Transformer-Grammars-Augmenting-Transformer) and its [official implementation](https://github.com/google-deepmind/transformer_grammars).

## Environment Setup
This code is tested with Python 3.7.17

### TG codebase
1. Create a virtual environment:
    ```bash
    cd src/syntactic_attention_based_metric_transformer_grammars
    python -m venv .tgenv
    source .tgenv/bin/activate
    ```
2. Install the packages:
    ```bash
    ./install.sh
    cd ../..
    deactivate
    ```

### RNNG codebase
Only needed for the beam-search condition.
1. Create a virtual environment:
    ```bash
    python -m venv .rnngenv
    source .rnngenv/bin/activate
    ```
2. Install the packages:
    ```bash
    pip install torch==1.12.1 sentencepiece tensorboard tqdm numpy nltk
    deactivate
    ```

### Other procedures
1. Create a virtual environment:
    ```bash
    python -m venv .venv
    .venv/bin/activate
    pip install nltk numpy pandas
    deactivate
    ```

### Data
1. Place the training data  (BLLIP-LG) as `train.txt`, `valid.txt`, and `text.txt` in `data/bllip-lg` directory.
2. Fetch the submodules under the `src/` directory, which hold the inference data (Natural Stories) and the RNNG used for the beam-search condition:
    ```bash
    git submodule update --init --recursive
    ```

## Train
### TG
1. Preprocess:
    ```bash
    bash scripts/preprocess_tg.sh
    ```
2. Train:
    ```bash
    bash scripts/train_tg.sh
    ```

### Transformer
1. Preprocess:
    ```bash
    bash scripts/preprocess_txl.sh
    ```
2. Train:
    ```bash
    bash scripts/train_txl.sh
    ```

### TG-comp
1. Train; use the same data as TG:
    ```bash
    bash scripts/train_txl_tree.sh
    ```

### RNNG
Only needed for the beam-search condition, where it parses Natural Stories.
1. Preprocess:
    ```bash
    bash scripts/preprocess_rnng.sh
    ```
2. Train:
    ```bash
    bash scripts/train_rnng.sh
    ```

## Compute NAE
The NAE can be computed over the gold trees of Natural Stories, or over the
parses that a word-synchronous beam search with the RNNG leaves in the beam at
each word, which is the parallel parsing experiment of Appendix D.
### TG
#### Gold trees
1. Preprocess:
    ```bash
    bash scripts/preprocess_inference_tg_gold.sh
    ```
2. Inference:
    ```bash
    bash scripts/inference_tg_gold.sh
    ```
3. Postprocess:
    ```bash
    bash scripts/postprocess_inference_tg_gold.sh
    ```

#### Beam search
1. Preprocess; parses Natural Stories with the trained RNNG:
    ```bash
    bash scripts/preprocess_inference_tg_bs.sh
    ```
2. Inference:
    ```bash
    bash scripts/inference_tg_bs.sh
    ```
3. Postprocess:
    ```bash
    bash scripts/postprocess_inference_tg_bs.sh
    ```

### Transformer
1. Preprocess:
    ```bash
    bash scripts/preprocess_inference_txl.sh
    ```
2. Inference:
    ```bash
    bash scripts/inference_txl.sh
    ```
3. Postprocess; use id/word mapping file from TG's preprocess:
    ```bash
    bash scripts/postprocess_inference_txl.sh
    ```

### TG-comp
#### Gold trees
1. Inference:
    ```bash
    bash scripts/inference_txl_tree_gold.sh
    ```
2. Postprocess; use id/word mapping file from TG's preprocess:
    ```bash
    bash scripts/postprocess_inference_txl_tree_gold.sh
    ```

#### Beam search
Uses the parses produced by TG's preprocess above.
1. Inference:
    ```bash
    bash scripts/inference_txl_tree_bs.sh
    ```
2. Postprocess; use id/word mapping file from TG's preprocess:
    ```bash
    bash scripts/postprocess_inference_txl_tree_bs.sh
    ```

## Precomputed NAE
The NAE values reported in the paper are included in `outputs/`, so that they can be used as predictors without re-running the training and the inference.

```
outputs/
├── seed=0/
├── seed=123/
└── seed=1234/
    ├── txl.csv            Transformer
    ├── tg_gold.csv        TG, gold trees
    ├── tg_bs.csv          TG, beam search
    ├── txl_tree_gold.csv  TG-comp, gold trees
    └── txl_tree_bs.csv    TG-comp, beam search
```

The three directories correspond to the three training seeds used in the paper. The `seed=42` that appears in the scripts is a placeholder; edit the paths in the scripts to compute the metrics under a different seed.

`outputs/nae_predictors.csv` joins all of these files on `id` and turns the log probabilities into surprisals, so that the predictors can be read in one go. Its columns are named `{condition}_{surp,nae,stack}_seed{seed}`, plus a `_mean` column per metric holding the mean over the three seeds, taken on the raw values. The table carries the projected NAE only; read the per-condition files below for the raw attention NAE. Regenerate it with `python ./src/merge_predictors.py`.

### Columns
Every file has one row per Natural Stories token, 10,256 rows in the same order, so the files can be joined on `id`.

| Column | Description |
| --- | --- |
| `id` | Token id of Natural Stories, in the `story.zone` format |
| `word` | The token itself |
| `sum_log_prob` | Natural-log probability of the token (in nats, negative). Surprisal is `-sum_log_prob` |
| `sum_original_metrics_nae` | NAE over the raw attention weights, summed over the attention heads |
| `sum_projected_metrics_nae` | NAE over the attention weights reweighted by the norm of each retrieved value vector after the output projection, summed over the attention heads |
| `sum_stack_count` | Number of elements on the stack of the incremental parse after the token. Absent from `txl.csv`, which involves no syntactic structure |

Every value is summed over the subword tokens and the words that constitute one Natural Stories zone, hence the `sum_` prefix.

**`sum_projected_metrics_nae` is the NAE reported in the paper**; `sum_original_metrics_nae` is not used in any of the analyses. The two correlate at r ≈ 0.97–1.00, so picking the wrong one yields results that look close to, but do not reproduce, the paper. In the beam-search files, the NAE and the stack count are averaged over the parses left in the beam, weighted by the softmax of their sequence log probabilities, whereas `sum_log_prob` is carried over from the highest-scoring parse in the beam rather than marginalized over it.

## Citation
```
@inproceedings{yoshida-etal-2025-attention,
    title = "If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?",
    author = "Yoshida, Ryo  and
      Isono, Shinnosuke  and
      Kajikawa, Kohei  and
      Someya, Taiga  and
      Sugimoto, Yushi  and
      Oseki, Yohei",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.483/",
    doi = "10.18653/v1/2025.acl-long.483",
    pages = "9795--9812",
    ISBN = "979-8-89176-251-0",
    abstract = "Recent work in computational psycholinguistics has revealed intriguing parallels between attention mechanisms and human memory retrieval, focusing primarily on vanilla Transformers that operate on token-level representations. However, computational psycholinguistic research has also established that syntactic structures provide compelling explanations for human sentence processing that token-level factors cannot fully account for. In this paper, we investigate whether the attention mechanism of Transformer Grammar (TG), which uniquely operates on syntactic structures as representational units, can serve as a cognitive model of human memory retrieval, using Normalized Attention Entropy (NAE) as a linking hypothesis between models and humans. Our experiments demonstrate that TG{'}s attention achieves superior predictive power for self-paced reading times compared to vanilla Transformer{'}s, with further analyses revealing independent contributions from both models. These findings suggest that human sentence processing involves dual memory representations{---}one based on syntactic structures and another on token sequences{---}with attention serving as the general memory retrieval algorithm, while highlighting the importance of incorporating syntactic structures as representational units."
}
```

## License
`src/syntactic_attention_based_metric_transformer_grammars/`: Apache License 2.0 (modified from [Transformer Grammars](https://github.com/google-deepmind/transformer_grammars)). See the directory's LICENSE file for details.

`src/syntactic_attention_based_metric_rnng-pytorch/`: MIT License (modified from [rnng-pytorch](https://github.com/aistairc/rnng-pytorch)). See the directory's LICENSE file for details.

`outputs/`: Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). These files are derived from the [Natural Stories Corpus](https://github.com/languageMIT/naturalstories), which is distributed under the same license.
