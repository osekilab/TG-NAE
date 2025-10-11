import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np


def print_log(message):
    """Format and output log messages"""
    print(f"INFO: {message}")


def extract_number(filename):
    """
    Extract the numeric part from a filename or directory name and return as integer
    Example: 'sent_123' -> 123, 'pointer_45' -> 45, 'sequence_67.csv' -> 67

    Args:
        filename: File name or directory name
    Returns:
        int: Extracted number
    """
    filename_str = str(filename)
    # Remove trailing .csv
    filename_str = filename_str.replace(".csv", "")
    # Get the number after the last underscore
    parts = filename_str.split("_")
    if len(parts) > 1:
        try:
            return int(parts[-1])
        except ValueError:
            print_log(f"Warning: Could not extract number from {filename}")
            return 0
    return 0


def count_stack_elements(words):
    """Calculate the number of stack elements"""
    stack = []
    for word in words:
        if word.startswith("("):
            stack.append("(")
        elif word.endswith(")"):
            while stack[-1] != "(":
                stack.pop()
            stack[-1] = "w"
        else:
            stack.append("w")
    return len(stack)


def process_sequence_file(file_path):
    """
    1つのsequence_*.csvファイルを処理する
    Returns: (sum_log_prob, word_info)
    最後の行がポインタ位置の単語情報となる
    """
    sum_log_prob = 0
    words = []
    last_row = None

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sum_log_prob += float(row["log_prob"])
            words.append(row["word"])
            if not row["word"].startswith("("):  # Exclude grammar tags starting with parentheses
                last_row = row

    if last_row is None:
        return sum_log_prob, None

    stack_count = count_stack_elements(words)

    word_info = {
        "word": last_row["word"],
        "log_prob": float(last_row["log_prob"]),
        "original_metrics_nae": float(last_row["original_metrics_nae"]),
        "projected_metrics_nae": float(last_row["projected_metrics_nae"]),
        "stack_count": stack_count,
    }

    return sum_log_prob, word_info


def process_pointer_directory(pointer_dir, use_weighted_nae=True):
    """
    Process all sequence_*.csv files in a pointer_* directory
    Args:
        pointer_dir: Path to pointer directory
        use_weighted_nae: Whether to use weighted average for NAE metrics
    Returns: Word information at the pointer position
    """
    sequence_results = []

    # Process all sequence_*.csv files in numerical order
    for sequence_file in sorted(
        Path(pointer_dir).glob("sequence_*.csv"), key=extract_number
    ):
        sum_log_prob, word_info = process_sequence_file(sequence_file)
        if word_info is not None:
            sequence_results.append((sum_log_prob, word_info))

    if not sequence_results:
        return None

    # Calculate log_probs and weights
    log_probs = np.array([result[0] for result in sequence_results])
    log_probs_normalized = log_probs - np.max(log_probs)
    probs = np.exp(log_probs_normalized)
    weights = probs / np.sum(probs)

    if use_weighted_nae:
        # Calculate NAE metrics with weighted average
        original_metrics_nae = np.sum(
            weights
            * np.array(
                [result[1]["original_metrics_nae"] for result in sequence_results]
            )
        )
        projected_metrics_nae = np.sum(
            weights
            * np.array(
                [result[1]["projected_metrics_nae"] for result in sequence_results]
            )
        )
        stack_count = np.sum(
            weights
            * np.array([result[1]["stack_count"] for result in sequence_results])
        )
    else:
        # Calculate NAE metrics with simple sum
        original_metrics_nae = np.sum(
            [result[1]["original_metrics_nae"] for result in sequence_results]
        )
        projected_metrics_nae = np.sum(
            [result[1]["projected_metrics_nae"] for result in sequence_results]
        )
        stack_count = np.sum([result[1]["stack_count"] for result in sequence_results])

    # Create result
    weighted_info = {
        "word": sequence_results[0][1]["word"],  # All words should be the same
        "log_prob": sequence_results[0][1]["log_prob"],  # log_prob value from a single row
        "original_metrics_nae": original_metrics_nae,
        "projected_metrics_nae": projected_metrics_nae,
        "stack_count": stack_count,
    }
    return weighted_info


def process_sent_directory(sent_dir, output_file, use_weighted_nae=True):
    """
    Process a sent_* directory and output the results to a CSV file
    Args:
        sent_dir: Path to input directory
        output_file: Path to output file
        use_weighted_nae: Whether to use weighted average for NAE metrics
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Process all pointer_* directories in numerical order and collect word information
    word_sequence_dict = {}  # Dictionary with pointer numbers as keys
    for pointer_dir in Path(sent_dir).glob("pointer_*"):
        pointer_num = extract_number(pointer_dir)
        word_info = process_pointer_directory(pointer_dir, use_weighted_nae)
        if word_info is not None:
            word_sequence_dict[pointer_num] = word_info

    # Sort by number and convert to list
    word_sequence = [word_sequence_dict[k] for k in sorted(word_sequence_dict.keys())]

    if not word_sequence:
        return

    # Output to CSV file
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "word",
                "log_prob",
                "original_metrics_nae",
                "projected_metrics_nae",
                "stack_count",
            ]
        )

        for word_info in word_sequence:
            writer.writerow(
                [
                    word_info["word"],
                    word_info["log_prob"],
                    word_info["original_metrics_nae"],
                    word_info["projected_metrics_nae"],
                    word_info["stack_count"],
                ]
            )


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process sequence files and generate aggregated metrics."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing sent_* directories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for sequence_*.csv files",
    )
    parser.add_argument(
        "--use-weighted-nae",
        action="store_true",
        help="Use probability-weighted average for NAE metrics (default: use simple values)",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Convert paths to Path objects
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Check if input directory exists
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Display processing method
    print(
        f"NAE metrics calculation: {'weighted average' if args.use_weighted_nae else 'simple values'}"
    )

    # Process all sent_* directories in numerical order
    sent_dirs = sorted(input_root.glob("sent_*"), key=extract_number)
    total_dirs = len(list(input_root.glob("sent_*")))

    for i, sent_dir in enumerate(sent_dirs, 1):
        sent_num = extract_number(sent_dir)
        output_file = output_root / f"sequence_{sent_num}.csv"
        process_sent_directory(sent_dir, output_file, args.use_weighted_nae)
        print_log(
            f"Processing directory {i}/{total_dirs}: sent_{sent_num:03d} -> sequence_{sent_num:03d}.csv"
        )


if __name__ == "__main__":
    main()
