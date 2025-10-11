import argparse
import re
from typing import List, Optional, Tuple, Union

import nltk
from tqdm import tqdm

# Define preserve_tags at module level
preserve_tags = {
    "-RRB-",
    "-LRB-",  # Round brackets
    "-RSB-",
    "-LSB-",  # Square brackets
    "-RCB-",
    "-LCB-",  # Curly brackets
    "-NONE-",  # Trace markers
    "-",
    "--",  # Hyphens
    "=",  # Equals sign
    ",",
    ".",
    ":",
    ";",
    "!",
    "?",  # Punctuation marks
    "``",
    "''",  # Quotes
}


def has_terminal(tree: nltk.Tree) -> bool:
    """
    Check if the tree has any terminal nodes (excluding traces).

    Args:
        tree (nltk.Tree): Tree to check

    Returns:
        bool: True if the tree has any terminals
    """
    if not isinstance(tree, nltk.Tree):
        return True  # A non-tree node is itself a terminal

    if is_trace(tree):
        return False

    return any(has_terminal(child) for child in tree)


def clean_label(label: str) -> str:
    """
    Clean a node label by removing function tags and indices while preserving special POS tags.

    Args:
        label (str): Original node label

    Returns:
        str: Cleaned label
    """
    # If the label is a special tag, preserve it
    if label in preserve_tags:
        return label

    # Extract the base category (before any function tags or index)
    base = label.split("-")[0]

    # If the base is empty, return the original label
    return base if base else label


def is_trace(tree: nltk.Tree) -> bool:
    """
    Check if the given tree is a trace node.

    Args:
        tree (nltk.Tree): Tree to check

    Returns:
        bool: True if the tree is a trace node
    """
    if not isinstance(tree, nltk.Tree):
        return False
    if len(tree) == 0:
        return False

    # Check if the last child is a -NONE- node
    last_child = tree[-1]
    if isinstance(last_child, nltk.Tree) and last_child.label() == "-NONE-":
        return True

    # Check for empty categories
    if isinstance(tree, nltk.Tree) and tree.label() == "-NONE-":
        return True

    return False


def remove_traces(tree: nltk.Tree) -> Optional[nltk.Tree]:
    """
    Remove trace elements and clean node labels in an NLTK Tree.
    Also removes nodes that have no terminals after trace removal.

    Args:
        tree (nltk.Tree): Input parse tree

    Returns:
        Optional[nltk.Tree]: Cleaned tree without traces and empty nodes
    """
    if not isinstance(tree, nltk.Tree):
        return tree

    if is_trace(tree):
        return None

    # Clean the current node's label
    clean_current_label = clean_label(tree.label())

    # Process children
    new_children: List[Union[str, nltk.Tree]] = []
    for child in tree:
        processed_child = remove_traces(child)
        if processed_child is not None and (
            not isinstance(processed_child, nltk.Tree) or has_terminal(processed_child)
        ):
            new_children.append(processed_child)

    # If no children remain after processing or no terminals exist in subtree, return None
    if not new_children or (isinstance(tree, nltk.Tree) and not has_terminal(tree)):
        return None

    return nltk.Tree(clean_current_label, new_children)


def preprocess_tree_string(tree_str: str) -> str:
    """
    Preprocess the tree string to handle common issues in Penn Treebank format.

    Args:
        tree_str (str): Original tree string

    Returns:
        str: Preprocessed tree string
    """
    # Normalize whitespace
    tree_str = " ".join(tree_str.split())

    # Ensure proper ROOT wrapping
    if not tree_str.startswith("("):
        tree_str = f"({tree_str})"

    # Balance parentheses
    open_count = tree_str.count("(")
    close_count = tree_str.count(")")
    if open_count > close_count:
        tree_str += ")" * (open_count - close_count)

    # Handle escaped characters
    tree_str = tree_str.replace('\\"', '"')

    return tree_str


def process_tree_with_error_handling(
    tree_str: str, error_handling: str = "keep"
) -> Tuple[str, bool]:
    """
    Process a tree with comprehensive error handling.

    Args:
        tree_str (str): Tree string to process
        error_handling (str): Error handling mode ('keep', 'skip', or 'mark')

    Returns:
        tuple[str, bool]: Processed tree string and success flag
    """
    try:
        # Preprocess the tree string
        processed_str = preprocess_tree_string(tree_str)

        # Parse the tree
        try:
            tree = nltk.Tree.fromstring(processed_str)
        except ValueError as e:
            # Try to handle common parsing issues
            if "multiple roots" in str(e):
                processed_str = f"(ROOT {processed_str})"
                tree = nltk.Tree.fromstring(processed_str)
            else:
                raise

        # Remove traces
        clean_tree = remove_traces(tree)
        if clean_tree is None:
            return None if error_handling == "skip" else tree_str, False

        # Format the output
        result = " ".join(str(clean_tree).split())
        while result.startswith("((") and result.endswith("))"):
            result = result[1:-1]

        return result, True

    except Exception as e:
        print(f"Warning: Failed to process tree: {e}")
        print(f"Problematic tree: {tree_str[:100]}...")

        if error_handling == "skip":
            return None, False
        elif error_handling == "mark":
            return f"ERROR_TREE:{tree_str}", False
        else:  # 'keep'
            return tree_str, False


def process_treebank_text(text: str) -> List[str]:
    """
    Process Penn Treebank format text and split into individual trees.

    Args:
        text (str): Input text in Penn Treebank format

    Returns:
        List[str]: List of individual tree strings
    """
    sentences = []
    current_sentence = ""
    parentheses_count = 0

    for char in text:
        current_sentence += char
        if char == "(":
            parentheses_count += 1
        elif char == ")":
            parentheses_count -= 1
            if parentheses_count == 0 and current_sentence.strip():
                # Clean up whitespace and newlines
                cleaned_sentence = " ".join(
                    line.strip() for line in current_sentence.strip().split("\n")
                )

                # Handle ROOT tag
                if cleaned_sentence.startswith("(ROOT "):
                    inner_sentence = cleaned_sentence[6:-1].strip()
                    sentences.append(inner_sentence)
                else:
                    sentences.append(cleaned_sentence)

                current_sentence = ""

    return sentences


def main():
    """
    Main function to process Penn Treebank files.
    """
    parser = argparse.ArgumentParser(
        description="Process Penn Treebank files and remove traces"
    )
    parser.add_argument(
        "--input_path", required=True, help="Path to Penn Treebank input file"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save processed trees"
    )
    parser.add_argument(
        "--error_handling",
        choices=["keep", "skip", "mark"],
        default="keep",
        help="How to handle failed trees: 'keep' (keep original), 'skip' (exclude), or 'mark' (prefix with ERROR_TREE:)",
    )
    parser.add_argument(
        "--failed_trees_path", help="Optional path to save failed trees separately"
    )
    args = parser.parse_args()

    # Read input file
    print(f"Reading file: {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Process treebank text to get individual trees
    print("Splitting into sentences...")
    trees = process_treebank_text(text)

    # Remove traces from each tree
    print("Removing traces...")
    processed_trees = []
    failed_trees = []

    for tree in tqdm(trees, desc="Cleaning trees"):
        result, success = process_tree_with_error_handling(tree, args.error_handling)
        if not success:
            failed_trees.append(tree)
        if result is not None:
            processed_trees.append(result)

    # Write results
    print(f"Writing results to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for tree in processed_trees:
            f.write(tree + "\n")

    # Write failed trees if path is provided
    if args.failed_trees_path and failed_trees:
        print(f"Writing {len(failed_trees)} failed trees to: {args.failed_trees_path}")
        with open(args.failed_trees_path, "w", encoding="utf-8") as f:
            for tree in failed_trees:
                f.write(tree + "\n")

    # Print summary
    total = len(trees)
    failed = len(failed_trees)
    success_rate = ((total - failed) / total) * 100 if total > 0 else 0

    print("\nProcessing complete!")
    print(f"Total trees: {total}")
    print(f"Successfully processed: {total - failed} ({success_rate:.1f}%)")
    print(f"Failed: {failed} ({100 - success_rate:.1f}%)")


if __name__ == "__main__":
    main()
