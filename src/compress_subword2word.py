import argparse
import csv
import json
from itertools import chain
from typing import Any, Dict, List, Tuple


def is_boundary(token: str) -> bool:
    """
    Check if the token represents a word boundary
    Args:
        token: Input token to check
    Returns:
        bool: True if token starts with '▁', False otherwise
    """
    return token.startswith("▁") or token.startswith("(") or token.endswith(")")


def process_tokens(
    tokens: List[Dict[str, Any]]
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Process tokens and aggregate metrics at word level
    Args:
        tokens: List of token dictionaries containing metrics
    Returns:
        Tuple of lists containing words and their metrics
    """
    all_words = []
    all_word_log_probs = []
    all_word_original_naes = []
    all_word_projected_naes = []

    current_word = tokens[0]["label_token"].replace("▁", "")
    current_word_log_prob = tokens[0]["log_prob"]
    current_word_original_nae = 0.0
    current_word_projected_nae = 0.0

    for token in tokens[1 : len(tokens) - 1]:
        label_token = token["label_token"]
        if label_token is None:
            current_word_original_nae += token["original_metrics"]["nae"]
            current_word_projected_nae += token["projected_metrics"]["nae"]

            all_words.append(current_word)
            all_word_log_probs.append(current_word_log_prob)
            all_word_original_naes.append(current_word_original_nae)
            all_word_projected_naes.append(current_word_projected_nae)

            current_word = ""
            current_word_log_prob = 0.0

            current_word_original_nae = 0.0
            current_word_projected_nae = 0.0
            continue

        if is_boundary(label_token):
            current_word_original_nae += token["original_metrics"]["nae"]
            current_word_projected_nae += token["projected_metrics"]["nae"]

            all_words.append(current_word)
            all_word_log_probs.append(current_word_log_prob)
            all_word_original_naes.append(current_word_original_nae)
            all_word_projected_naes.append(current_word_projected_nae)

            if label_token:
                current_word = label_token.replace("▁", "")
            else:
                current_word = None
            current_word_log_prob = token["log_prob"]

            current_word_original_nae = 0.0
            current_word_projected_nae = 0.0
            continue

        current_word += label_token
        current_word_log_prob += token["log_prob"]
        current_word_original_nae += token["original_metrics"]["nae"]
        current_word_projected_nae += token["projected_metrics"]["nae"]

    current_word_original_nae += tokens[-1]["original_metrics"]["nae"]
    current_word_projected_nae += tokens[-1]["projected_metrics"]["nae"]

    all_words.append(current_word)
    all_word_log_probs.append(current_word_log_prob)
    all_word_original_naes.append(current_word_original_nae)
    all_word_projected_naes.append(current_word_projected_nae)

    assert (
        len(all_words)
        == len(all_word_log_probs)
        == len(all_word_original_naes)
        == len(all_word_projected_naes)
    )
    return (
        all_words,
        all_word_log_probs,
        all_word_original_naes,
        all_word_projected_naes,
    )


def process_json_file(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process JSON data and aggregate tokens into words with metrics
    Args:
        json_data: Input JSON data containing tokens and metrics
    Returns:
        List[Dict[str, Any]]: Processed results with words and aggregated metrics
    """
    all_tokens = list(
        chain.from_iterable(chunk["tokens"] for chunk in json_data["chunks"])
    )
    words, log_probs, original_naes, projected_naes = process_tokens(all_tokens)

    return [
        {
            "word": word,
            "log_prob": log_prob,
            "original_metrics_nae": original_nae,
            "projected_metrics_nae": projected_nae,
        }
        for word, log_prob, original_nae, projected_nae in zip(
            words, log_probs, original_naes, projected_naes
        )
    ]


def write_csv(results: List[Dict[str, Any]], output_file: str):
    """
    Write results to CSV file
    Args:
        results: List of dictionaries containing words and metrics
        output_file: Path to output CSV file
    """
    if not results:
        return

    fieldnames = ["word", "log_prob", "original_metrics_nae", "projected_metrics_nae"]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_arguments():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Aggregate token-level metrics to word-level metrics"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    return parser.parse_args()


def main():
    """
    Main function to run the token aggregation process
    """
    args = parse_arguments()

    with open(args.input, "r") as f:
        json_data = json.load(f)

    results = process_json_file(json_data)
    write_csv(results, args.output)


if __name__ == "__main__":
    main()
