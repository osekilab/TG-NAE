import argparse
from pathlib import Path

import nltk
from nltk.tree import Tree
from tqdm import tqdm


def extract_terminals(input_file, output_file):
    """
    Parse trees from input file and extract only terminal nodes to output file.
    Args:
        input_file (str or Path): Path to input file containing parse trees
        output_file (str or Path): Path where terminal nodes will be written
    """
    # Convert to Path objects
    input_path = Path(input_file)
    output_path = Path(output_file)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_path}")
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:

        for line in tqdm(f_in):
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                # Parse the tree string
                tree = Tree.fromstring(line)

                # Extract only terminal nodes (leaves)
                terminals = tree.leaves()

                # Join terminals with spaces and write to output
                f_out.write(" ".join(terminals) + "\n")

            except Exception as e:
                print(f"Error processing line in {input_path}: {e}")
                continue

    print(f"Created {output_path}")


def setup_parser():
    """
    Set up command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Extract terminal nodes from parse trees in text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input file paths
    parser.add_argument(
        "--train-input", type=str, help="Path to training data parse trees"
    )
    parser.add_argument(
        "--valid-input", type=str, help="Path to validation data parse trees"
    )
    parser.add_argument("--test-input", type=str, help="Path to test data parse trees")

    # Output file paths
    parser.add_argument(
        "--train-output", type=str, help="Path for training data terminals"
    )
    parser.add_argument(
        "--valid-output", type=str, help="Path for validation data terminals"
    )
    parser.add_argument("--test-output", type=str, help="Path for test data terminals")

    # Optional: Process only specific files
    parser.add_argument(
        "--only",
        choices=["train", "valid", "test"],
        nargs="+",
        help="Process only specific file types",
    )

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Create input and output path mappings
    input_paths = {
        "train": args.train_input,
        "valid": args.valid_input,
        "test": args.test_input,
    }

    output_paths = {
        "train": args.train_output,
        "valid": args.valid_output,
        "test": args.test_output,
    }

    # Filter files to process if --only is specified
    if args.only:
        input_paths = {k: v for k, v in input_paths.items() if k in args.only}
        output_paths = {k: v for k, v in output_paths.items() if k in args.only}

    # Validate paths
    for file_type, input_path in input_paths.items():
        if input_path is None:
            print(f"Warning: No input path specified for {file_type}")
            continue
        if output_paths[file_type] is None:
            print(f"Warning: No output path specified for {file_type}")
            continue

        extract_terminals(input_path, output_paths[file_type])


if __name__ == "__main__":
    main()
