#!/usr/bin/env python3
import argparse
import os


def process_line(line):
    tokens = line.split()
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i][-1] != ")":
            break
    return " ".join(tokens[: i + 1])


def process_file(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    processed_lines = [process_line(line.strip()) for line in lines]

    with open(output_file, "w") as f:
        for line in processed_lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove labeled closing brackets from parse trees"
    )
    parser.add_argument("--input", required=True, help="Input file to process")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process the file
    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
