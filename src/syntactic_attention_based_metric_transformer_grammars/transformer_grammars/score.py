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

"""Score tokenized post-processed sequences.

Note: This is a naive implementation with batch size 1 for simplicity. It can be
extended to support > 1 batch sizes, or returning activations from the model.
"""

import datetime
import functools
import json
from itertools import tee
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from transformer_grammars import common
from transformer_grammars.data import preprocessing, text_dataset
from transformer_grammars.data import tokenizer_utils as utils
from transformer_grammars.training import checkpoint


def _sequences_iterator(fname, add_eos):
    """Iterator over the pretokenized dataset."""
    ds = text_dataset.PreEncodedTextDataset(
        filename=fname, num_samples=None, add_bos=True, add_eos=add_eos
    )
    ds = ds.raw_dataset(
        shuffle=False,
        shuffle_buffer=None,
        sample_without_replacement=False,
        num_epochs=1,
    )
    return ds.as_numpy_iterator()


@functools.partial(jax.jit, static_argnums=(0, 1))
def _call_model(forward, maskrules, params, state, chunk):
    """Calls the model for scoring purposes and returns attention weights.

    Args:
        forward: Forward function from build_forward
        maskrules: Mask rules object
        params: Model parameters
        state: Model state
        chunk: Input chunk containing sequences and labels

    Returns:
        A tuple containing:
            - A tuple of model outputs:
                - log_probs: Log probabilities for all tokens
                - labels_log_probs: Log probabilities for the actual labels
                - chunk_log_prob: Sum of log probabilities for the chunk
                - attention_info: Dictionary containing:
                    - weights: Dictionary with structure:
                        {
                            'restricted': {
                                layer_idx: {
                                    'attention_weights': weights,
                                    'value_heads': values`
                                }
                            },
                            'unrestricted': {
                                layer_idx: {
                                    'attention_weights': weights,
                                    'value_heads': values`
                                }
                            }
                        }
                        where weights have shape [H, T, T+M] and
                        values have shape [H, T, T+M, H*V]
                    - layer_outputs: Layer outputs tensor
            - Updated model state
    """
    model_inputs = common.model_input_from_chunk(chunk, maskrules)
    (logits, layer_outputs, attention_weights), state = forward(
        params, state, rng=None, **model_inputs
    )

    # Calculate log probabilities
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    mask = jnp.logical_and(
        jnp.greater(chunk.labels, 0), jnp.greater(chunk.seq_idx, -1)[:, None]
    ).astype(jnp.int32)
    log_probs *= mask[:, :, None]

    # Calculate label-specific log probabilities
    labels_log_probs = jax.vmap(jax.vmap(lambda t, idx: t[idx], 0, 0), 0, 0)(
        log_probs, chunk.labels
    )
    chunk_log_prob = jnp.sum(labels_log_probs, axis=1)

    # Process the main outputs with batch dimension removal
    log_probs = jnp.squeeze(log_probs, axis=0)
    labels_log_probs = jnp.squeeze(labels_log_probs, axis=0)
    chunk_log_prob = jnp.squeeze(chunk_log_prob, axis=0)
    layer_outputs = jnp.squeeze(layer_outputs, axis=0)

    # Process attention weights and value heads separately by layer type
    squeezed_attention_weights = {}
    for layer_type, layer_weights in attention_weights.items():
        squeezed_attention_weights[layer_type] = {}
        for layer_idx, weights_dict in layer_weights.items():
            # Handle both original weights and projected values
            squeezed_attention_weights[layer_type][layer_idx] = {
                "attention_weights": jnp.squeeze(
                    weights_dict["attention_weights"], axis=0
                ),
                "value_heads": jnp.squeeze(weights_dict["value_heads"], axis=0),
            }

    attention_info = {
        "weights": squeezed_attention_weights,
        "layer_outputs": layer_outputs,
    }

    return (log_probs, labels_log_probs, chunk_log_prob, attention_info), state


def convert_to_serializable(obj):
    """Convert JAX arrays, numpy arrays and other non-serializable objects to lists."""
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        return np.asarray(obj).tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, (np.int32, np.int64, jnp.int32, jnp.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64, jnp.float32, jnp.float64)):
        return float(obj)
    return obj


def compute_projected_values(
    attention_weights: jnp.ndarray,  # [H, Q, K]
    value_heads: jnp.ndarray,  # [K, H, V]
    linear_weights: jnp.ndarray,  # [H, V, H*V]
    linear_bias: jnp.ndarray,  # [H*V]
) -> jnp.ndarray:
    # Transform value_heads to match PyTorch implementation
    # [K, H, V] -> [K, H, 1, V]
    seq_len, num_heads, value_size = value_heads.shape
    value_layer = jnp.expand_dims(value_heads, axis=2)

    # Reshape final linear weights to match PyTorch implementation
    output_weight = linear_weights.reshape(
        num_heads, value_size, num_heads * value_size
    )

    # Transform values using output weights
    # [K, H, 1, V] @ [H, V, H*V] -> [K, H, H*V]
    projected_states = jnp.matmul(value_layer, output_weight).squeeze(2)

    # Reorder dimensions
    # [K, H, H*V] -> [H, K, H*V]
    projected_states = jnp.transpose(projected_states, (1, 0, 2))

    # Add bias
    bias = linear_bias / num_heads
    projected_states = projected_states + bias

    # Final transformation using attention weights
    # [H, Q, K] and [H, K, H*V] -> [H, Q, K, H*V]
    final_states = jnp.einsum("hqk,hkd->hqkd", attention_weights, projected_states)

    return final_states


def calculate_metrics_per_token_xl(
    attention_weights,
    memory_length,
    final_linear_weights,
    final_linear_bias,
):
    """Calculate NAE, ΔNAE, MD metrics, and valid elements for each token position.

    Args:
        attention_weights: Dictionary containing:
            - 'original': JAX array of shape [H, T, M+T]
            - 'projected': JAX array of shape [H, T, M+T, H*V]
        memory_length: Integer indicating the length of memory context

    Returns:
        Dictionary containing metrics and valid elements for each token position and type
    """
    metrics = {"original": {}, "projected": {}}

    for weight_type in ["attention_weights", "value_heads"]:
        if weight_type == "value_heads":
            value_heads = attention_weights[weight_type]  # [H, T, M+T, H*V]
            # Calculate projected values
            projected_values = compute_projected_values(
                attention_weights["attention_weights"],
                value_heads,
                final_linear_weights,
                final_linear_bias,
            )
            # Calculate norms of projected values
            weights = jnp.linalg.norm(projected_values, axis=-1)  # [H, T, M+T]
            # Normalize by sum of norms
            weights_sum = jnp.sum(weights, axis=-1, keepdims=True)  # [H, T, 1]
            weights = jnp.where(weights_sum > 0, weights / weights_sum, 0.0)
        else:
            weights = attention_weights[
                weight_type
            ]  # [H, T, M+T]; in scoreing, batch size is 1
        num_heads, seq_len, total_len = weights.shape

        context_lengths = jnp.minimum(memory_length + jnp.arange(seq_len), total_len)

        base_mask = (
            jnp.arange(total_len)[None, :] < context_lengths[:, None]
        )  # [T, M+T]

        nonzero_mask = weights > 0  # [H, T, M+T]
        any_head_attends = jnp.any(nonzero_mask, axis=0)  # [T, M+T]

        mask = (base_mask & any_head_attends).astype(jnp.float32)

        # normalize weights
        weights_masked = weights * mask[None, :, :]
        weights_sum = jnp.sum(weights_masked, axis=-1, keepdims=True)
        normalized_weights = jnp.where(
            weights_sum > 0, weights_masked / weights_sum, 0.0
        )
        normalized_weights = normalized_weights * mask[None, :, :]

        # calculate NAE
        entropy = -jnp.sum(
            normalized_weights
            * jnp.log2(normalized_weights + 1e-10)
            * mask[None, :, :],
            axis=-1,
        )  # [H, T]
        valid_elements = jnp.sum(mask, axis=-1)  # [T]
        max_entropy = jnp.log2(valid_elements)  # [T]
        nae = jnp.where(
            valid_elements > 1,
            entropy / max_entropy[None, :],
            0.0,
        )  # [H, T]

        if seq_len > 1:
            # |NAE(t) - NAE(t-1)|
            prev_nae = jnp.roll(nae, 1, axis=1)  # [H, T]
            delta_nae = jnp.abs(nae - prev_nae)  # [H, T]
            # set delta_nae for the first token to 0
            delta_nae = delta_nae.at[:, 0].set(0.0)

            # ||W(t) - W(t-1)||_1
            context_lengths_for_md = jnp.minimum(
                memory_length + jnp.arange(1, seq_len + 1), total_len
            )  # [T]
            base_mask_for_md = (
                jnp.arange(total_len)[None, :] < context_lengths_for_md[:, None]
            )  # [T, M+T]
            nonzero_mask_for_md = weights > 0  # [H, T, M+T]
            any_head_attends_for_md = jnp.any(nonzero_mask_for_md, axis=0)  # [T, M+T]
            mask_for_md = (base_mask_for_md & any_head_attends_for_md).astype(
                jnp.float32
            )  # [T, M+T]

            weights_masked_for_md = weights * mask_for_md[None, :, :]  # [H, T, M+T]
            prev_weights_masked_for_md = jnp.roll(
                weights_masked_for_md, 1, axis=1
            )  # [H, T, M+T]
            md = jnp.sum(
                jnp.abs(weights_masked_for_md - prev_weights_masked_for_md), axis=-1
            )
            # set md for the first token to 0
            md = md.at[:, 0].set(0.0)

        else:
            delta_nae = jnp.zeros_like(nae)
            md = jnp.zeros_like(nae)

        # aggregate metrics for each token position
        key_name = "original" if weight_type == "attention_weights" else "projected"
        metrics[key_name]["nae"] = [float(x) for x in jnp.sum(nae, axis=0)]
        metrics[key_name]["delta_nae"] = [float(x) for x in jnp.sum(delta_nae, axis=0)]
        metrics[key_name]["md"] = [float(x) for x in jnp.sum(md, axis=0)]
        # save valid elements for each token position
        metrics[key_name]["valid_elements"] = [float(x) for x in valid_elements]

    return metrics


def main(
    tokenizer,
    checkpoint_path,
    input_,
    output_dir_,
    add_eos,
    _,
):
    """Score and save attention metrics to file for Transformer XL model."""
    # Extract value from flag handles
    tokenizer = tokenizer.value
    checkpoint_path = checkpoint_path.value
    input_ = input_.value
    add_eos = add_eos.value
    output_dir = output_dir_.value

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get dictionary and ranges
    dic, ranges = utils.get_dictionary_and_ranges(tokenizer)

    # Load checkpoint and build model
    ckpt = checkpoint.load_checkpoint(checkpoint_path)
    model_cfg = ckpt.config
    params = ckpt.params

    # Prepare model kwargs
    model_kwargs = dict(**model_cfg)
    model_kwargs.pop("extra_attention_mask_name", None)
    model_kwargs.pop("extra_attention_mask_kwargs", None)

    # Build maskrules and update model kwargs
    maskrules = common.build_maskrules(model_cfg)
    model_kwargs["num_attns"] = maskrules.num_attention_functions
    model_kwargs["vocab_size"] = ranges.vocab_size

    # Build forward function
    forward = common.build_forward(
        model_kwargs, maskrules, ranges, is_training=False
    ).apply

    # Get memory length from config
    memory_length = model_kwargs["memory_length"]

    # Get iterators
    sequences_it = _sequences_iterator(input_, add_eos)
    # Create two iterators: one for counting and one for processing
    sequences_it_count, sequences_it_process = tee(sequences_it)

    # Get chunks iterators
    chunks_it_count = preprocessing.get_chunks_from_dataset(
        sequences_it_count,
        maskrules,
        ranges,
        shape_prefix=(1,),
        multithread=False,
        use_monitor_thread=False,
    )

    # Count total chunks
    total_chunks = sum(1 for _ in chunks_it_count)

    # Get chunks for processing
    chunks_it = preprocessing.get_chunks_from_dataset(
        sequences_it_process,
        maskrules,
        ranges,
        shape_prefix=(1,),
        multithread=False,
        use_monitor_thread=False,
    )

    # Initialize counters and storage
    state = None
    seq_log_prob = 0.0
    total_log_prob = 0.0
    current_sequence = []
    sequence_counter = 0

    for chunk in tqdm(chunks_it, desc="Processing sequences", total=total_chunks):
        (log_probs, labels_log_probs, chunk_log_prob, attention_info), state = (
            _call_model(forward, maskrules, params, state, chunk)
        )

        inputs = chunk.inputs[0]
        labels = chunk.labels[0]
        seq_log_prob += chunk_log_prob
        total_log_prob += chunk_log_prob

        if chunk.beginning_of_seq.item():
            sequence_counter += 1
            current_sequence = []

        # Filter out padding tokens
        valid_tokens = [
            (inp, lab, lp)
            for inp, lab, lp in zip(inputs, labels, labels_log_probs)
            if inp != 0
        ]
        valid_token_positions = [i for i, (inp, _, _) in enumerate(valid_tokens)]

        # Get attention weights for each layer type
        chunk_metrics = None
        if attention_info.get("weights"):
            for layer_type, layers in attention_info["weights"].items():
                if layers:  # レイヤーが存在する場合
                    max_layer = max(layers.keys())
                    attention_weights = {
                        "attention_weights": layers[max_layer]["attention_weights"],
                        "value_heads": layers[max_layer]["value_heads"],
                    }
                    # Calculate metrics for each token position
                    chunk_metrics = calculate_metrics_per_token_xl(
                        attention_weights,
                        memory_length,
                        params[f"lm/~/core/h{max_layer}_attn/linear"]["w"],
                        params[f"lm/~/core/h{max_layer}_attn/linear"]["b"],
                    )
                    break  # 最初の有効なレイヤータイプのメトリクスを使用

        # Create chunk data with tokens and their corresponding metrics
        chunk_data = {
            "tokens": [
                {
                    "input_token": dic[int(inp)],
                    "label_token": dic[int(lab)] if lab != 0 else None,
                    "log_prob": float(lp) if lab != 0 else None,
                }
                for inp, lab, lp in valid_tokens
            ],
            "chunk_log_prob": float(chunk_log_prob),
        }

        # Add metrics for each token if available
        if chunk_metrics:
            for i, metrics_position in enumerate(valid_token_positions):
                for weight_type in ["original", "projected"]:
                    if metrics_position < len(chunk_metrics[weight_type]["nae"]):
                        token_metrics = {
                            "nae": chunk_metrics[weight_type]["nae"][metrics_position],
                            "delta_nae": chunk_metrics[weight_type]["delta_nae"][
                                metrics_position
                            ],
                            "md": chunk_metrics[weight_type]["md"][metrics_position],
                            "valid_elements": chunk_metrics[weight_type][
                                "valid_elements"
                            ][metrics_position],
                        }

                        chunk_data["tokens"][i][
                            f"{weight_type}_metrics"
                        ] = token_metrics

                # コンテキスト長を記録
                chunk_data["tokens"][i]["context_length"] = (
                    memory_length + metrics_position
                )

        current_sequence.append(chunk_data)

        if chunk.end_of_seq.item():
            sequence_file = output_dir / f"sequence_{sequence_counter}.json"
            sequence_data = {
                "sequence_log_prob": float(seq_log_prob),
                "memory_length": int(memory_length),
                "chunks": current_sequence,
            }
            with sequence_file.open("w") as f:
                json.dump(sequence_data, f, indent=2)
            seq_log_prob = 0.0

    # Save summary statistics
    summary_file = output_dir / "summary.json"
    summary_data = {
        "total_log_prob": float(total_log_prob),
        "num_sequences": sequence_counter,
        "memory_length": int(memory_length),
        "model_config": {k: str(v) for k, v in model_kwargs.items()},
        "checkpoint_path": str(checkpoint_path),
    }
    with summary_file.open("w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAnalysis complete. Results saved in: {output_dir}")
    return output_dir
