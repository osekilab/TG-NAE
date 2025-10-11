import argparse
import os
import re
from typing import List, Tuple


def count_required_brackets(tree: str, pos: int) -> int:
    """
    Calculate the difference between opening and closing brackets up to the specified position.
    """
    open_count = tree[:pos].count("(")
    close_count = tree[:pos].count(")")
    return open_count - close_count


def find_terminal_positions(tree: str) -> List[Tuple[int, int, str]]:
    """
    Get positions of all terminal nodes in the tree.
    Terminal nodes are nodes where actual words or symbols follow POS tags.
    returns: List of (start_pos, end_pos, terminal_text)
    """
    positions = []
    stack = []
    terminal_pattern = False
    current_start = -1
    pos = 0

    while pos < len(tree):
        if tree[pos] == "(":
            stack.append(pos)
            pos += 1
            continue

        if tree[pos] == ")":
            if stack:
                start = stack.pop()
                # Get the current bracketed part
                current_span = tree[start : pos + 1]
                # Check pattern: bracket, tag, space, terminal, bracket
                if " " in current_span:  # Space required between tag and terminal
                    tag_end = current_span.find(" ")
                    terminal = current_span[tag_end + 1 : -1].strip()
                    if (
                        terminal and not "(" in terminal and not ")" in terminal
                    ):  # Terminal should not contain brackets
                        terminal_start = start + tag_end + 1
                        terminal_end = pos
                        positions.append(
                            (terminal_start, terminal_end, terminal.strip())
                        )
            pos += 1
            continue

        pos += 1

    # Sort by position
    return sorted(positions, key=lambda x: x[0])


def get_partial_tree(tree: str, terminal_pos: Tuple[int, int, str]) -> str:
    """
    Generate a partial tree up to the specified terminal position and add necessary closing brackets.
    """
    # Get substring up to the end position of the terminal element
    partial = tree[: terminal_pos[1]]

    # Calculate the number of required closing brackets
    required_brackets = count_required_brackets(partial, terminal_pos[1])

    # Add necessary closing brackets
    partial += ")" * required_brackets

    return partial


def process_tree_file(input_file: str, output_dir: str):
    """
    Process input file and save partial trees for each sentence to files.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        for sent_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Create directory for each sentence
            sent_dir = os.path.join(output_dir, f"sent_{sent_idx}")
            os.makedirs(sent_dir, exist_ok=True)

            # Get positions of terminal nodes
            terminal_positions = find_terminal_positions(line)

            # Generate and save partial tree for each terminal position
            for terminal_idx, terminal_pos in enumerate(terminal_positions):
                partial_tree = get_partial_tree(line, terminal_pos)

                # Save partial tree to file
                output_file = os.path.join(sent_dir, f"pointer_{terminal_idx}.txt")
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(partial_tree + "\n")


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script to generate and save partial trees from parse trees"
    )
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    print(f"Starting process...")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")

    # Process file
    process_tree_file(args.input, args.output_dir)

    print("Processing completed.")


if __name__ == "__main__":
    main()
