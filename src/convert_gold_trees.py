import os
import re
from typing import List, Tuple


def count_required_brackets(tree: str, pos: int) -> int:
    """
    Calculate the difference between the number of opening brackets and closing brackets up to the specified position
    """
    open_count = tree[:pos].count("(")
    close_count = tree[:pos].count(")")
    return open_count - close_count


def find_word_positions(tree: str) -> List[Tuple[int, int]]:
    """
    Get the start and end positions of each word in the tree
    """
    # Word pattern: non-whitespace characters following brackets and tags
    pattern = r"\([A-Z]+\s+([^\s()]+)\)"
    positions = []

    for match in re.finditer(pattern, tree):
        word_start = match.start(1)  # Start position of the word
        word_end = match.end(1)  # End position of the word
        positions.append((word_start, word_end))

    return positions


def get_partial_tree(tree: str, word_pos: Tuple[int, int]) -> str:
    """
    Generate a partial tree up to the specified word position and add necessary closing brackets
    """
    # Get substring up to the end of the word
    partial = tree[: word_pos[1]]

    # Calculate the number of required closing brackets
    required_brackets = count_required_brackets(partial, word_pos[1])

    # Add necessary closing brackets
    partial += ")" * required_brackets

    return partial


def process_tree_file(input_file: str, output_dir: str):
    """
    Process the input file and save partial trees for each sentence to files
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

            # Get word positions
            word_positions = find_word_positions(line)

            # Generate and save partial trees up to each word position
            for word_idx, word_pos in enumerate(word_positions):
                partial_tree = get_partial_tree(line, word_pos)

                # Save partial tree to file
                output_file = os.path.join(sent_dir, f"pointer_{word_idx}.txt")
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(partial_tree + "\n")


def main():
    # Usage example
    input_file = "input.txt"  # Path to input file
    output_dir = "output"  # Path to output directory

    process_tree_file(input_file, output_dir)


if __name__ == "__main__":
    main()
