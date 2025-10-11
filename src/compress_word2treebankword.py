import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_and_concatenate_sequences(sequence_files):
    """
    Read and concatenate all sequence files
    """
    print("Reading and concatenating sequence files...")
    all_sequences = []
    for file in tqdm(sequence_files, desc="Reading files"):
        try:
            df = pd.read_csv(file)
            print(f"Successfully read {file}: {len(df)} rows")  # Debug output
            all_sequences.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
            raise

    result = pd.concat(all_sequences, ignore_index=True)
    print(f"Total rows after concatenation: {len(result)}")  # Debug output
    return result


def read_id_mapping(id_file):
    """Read ID mapping file"""
    return pd.read_csv(id_file, dtype={"id": str})


def find_matching_rows(target_words, sequence_df, start_idx=0):
    """
    Search for rows corresponding to the given words while preserving order
    """
    matching_indices = []
    current_idx = start_idx

    for target in target_words:
        # Search for word from current position
        while current_idx < len(sequence_df):
            if sequence_df.iloc[current_idx]["word"] == target:
                matching_indices.append(current_idx)
                current_idx += 1
                break
            current_idx += 1

        if current_idx >= len(sequence_df):
            print(f"Warning: Reached end of sequence_df while searching for '{target}'")

    return matching_indices, current_idx


def aggregate_metrics(id_df, sequence_df):
    """Aggregate metrics"""
    print("Aggregating metrics...")
    print(f"Input sequence_df size: {len(sequence_df)}")  # Debug output

    # DataFrame to store results
    result_df = pd.DataFrame()
    result_df["id"] = id_df["id"]
    result_df["word"] = id_df["word"]

    # Metrics columns
    metrics = (
        [
            "log_prob",
            "original_metrics_nae",
            "projected_metrics_nae",
            "stack_count",
        ]
        if "stack_count" in sequence_df.columns
        else [
            "log_prob",
            "original_metrics_nae",
            "projected_metrics_nae",
        ]
    )
    for metric in metrics:
        result_df[f"sum_{metric}"] = 0.0

    # Process each ID group sequentially
    current_idx = 0
    for idx, row in tqdm(
        result_df.iterrows(), total=len(result_df), desc="Processing rows"
    ):
        target_words = row["word"].split()

        # Debug output for progress tracking
        if idx % 100 == 0:
            print(f"Processing ID {row['id']}, current_idx: {current_idx}")

        matching_indices, current_idx = find_matching_rows(
            target_words, sequence_df, start_idx=current_idx
        )

        # Verify number of matched rows
        if len(matching_indices) != len(target_words):
            print(
                f"Warning: ID {row['id']} - Expected {len(target_words)} matches, got {len(matching_indices)}"
            )

        for metric in metrics:
            metric_sum = (
                sequence_df.iloc[matching_indices][metric].sum()
                if matching_indices
                else 0.0
            )
            result_df.at[idx, f"sum_{metric}"] = metric_sum

    return result_df


def natural_sort_key(s):
    """Key function for sorting with numerical consideration"""
    import re

    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s

    return [tryint(c) for c in re.split("([0-9]+)", s)]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from sequence files based on sequential word matching"
    )
    parser.add_argument("--id_file", required=True, help="Path to the ID mapping file")
    parser.add_argument(
        "--sequence_dir", required=True, help="Directory containing sequence CSV files"
    )
    parser.add_argument("--output_file", required=True, help="Output file path")
    parser.add_argument(
        "--precision",
        type=int,
        default=15,
        help="Number of decimal places for floating point numbers",
    )

    args = parser.parse_args()

    try:
        # Read ID mapping file
        print(f"Reading ID mapping file: {args.id_file}")
        id_df = read_id_mapping(args.id_file)

        # Get list of sequence files
        sequence_files = glob.glob(f"{args.sequence_dir}/sequence_*.csv")
        if not sequence_files:
            raise ValueError(f"No sequence files found in {args.sequence_dir}")

        # Sort in natural order
        sequence_files.sort(key=natural_sort_key)
        print("Sorted files:", "\n".join(sequence_files))  # Debug output
        print(f"Found {len(sequence_files)} sequence files")

        # Concatenate all sequence files
        sequence_df = read_and_concatenate_sequences(sequence_files)

        # Aggregate metrics
        result_df = aggregate_metrics(id_df, sequence_df)

        # Save results
        print(f"Saving results to {args.output_file}")

        # Format settings for saving
        result_df.to_csv(
            args.output_file,
            index=False,
            float_format=f"%.{args.precision}f",
            na_rep="0",  # Output missing values as 0
            quoting=csv.QUOTE_MINIMAL,  # Use quotes only when necessary
        )

        print("Done!")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
