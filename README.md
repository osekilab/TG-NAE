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
2. Add the inference data (Natural Stories) as a submodule under `src/` directory:
    ```bash
    git submodule add https://github.com/languageMIT/naturalstories.git src/naturalstories
    ```

## Train
### TG
1. Preprocess:
    ```bash
    bash scripts/preprocess_train_tg.sh
    ```
2. Train:
    ```bash
    bash scripts/train_tg.sh
    ```

### Transformer
1. Preprocess:
    ```bash
    bash scripts/preprocess_train_txl.sh
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

## Compute NAE
### TG
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
1. Inference:
    ```bash
    bash scripts/inference_txl_tree_gold.sh
    ```
2. Postprocess; use id/word mapping file from TG's preprocess:
    ```bash
    bash scripts/postprocess_inference_txl_tree_gold.sh
    ```

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
