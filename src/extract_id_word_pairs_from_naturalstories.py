import argparse
import csv
from collections import defaultdict

import nltk
from nltk import Tree
from tqdm import tqdm


def extract_word_id_pairs(tree_str):
    # Create NLTK Tree object from string
    tree = Tree.fromstring(tree_str)

    # List to store word and ID pairs
    pairs = []

    def traverse_tree(t):
        if isinstance(t, Tree):
            for child in t:
                traverse_tree(child)
        else:
            # Extract word and ID from terminal node
            if "/" in str(t):
                word, id_str = str(t).split("/")
                # Process only items containing digits (exclude symbols, etc.)
                if any(c.isdigit() for c in id_str):
                    # Extract "X.X" part from ID
                    id_parts = id_str.split(".")
                    if len(id_parts) >= 2:
                        primary = int(id_parts[0])
                        secondary = int(id_parts[1])
                        base_id = f"{primary}.{secondary}"
                        pairs.append((base_id, word))

    # Traverse the tree
    traverse_tree(tree)
    return pairs


def aggregate_by_id(pairs):
    # Group by ID
    grouped_data = defaultdict(list)

    for base_id, word in pairs:
        grouped_data[base_id].append(word)

    # Format results
    result = []
    for base_id, words in grouped_data.items():
        combined_words = " ".join(words)
        result.append({"id": base_id, "word": combined_words})

    # Custom key function for comparing IDs as numbers
    def sort_key(item):
        primary, secondary = map(int, item["id"].split("."))
        return (primary, secondary)

    return sorted(result, key=sort_key)  # Sort IDs numerically


def process_parse_trees(content):
    print("Parsing trees...")
    # Split into individual trees to process multiple trees
    tree_strings = content.strip().split("\n(ROOT")
    tree_strings = [s if s.startswith("(ROOT") else "(ROOT" + s for s in tree_strings]

    all_pairs = []
    for tree_str in tqdm(tree_strings, desc="Processing trees"):
        if tree_str.strip():
            pairs = extract_word_id_pairs(tree_str)
            all_pairs.extend(pairs)

    print("Aggregating results...")
    # Group by ID and generate results
    return aggregate_by_id(all_pairs)


def save_results(results, output_file):
    # Save results to CSV file
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "word"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Parse tree structure and aggregate words by ID"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input file containing the parse tree structure",
    )
    parser.add_argument(
        "--output_file", required=True, help="Output CSV file to save the results"
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="Input file encoding (default: utf-8)"
    )
    args = parser.parse_args()

    try:
        # Read input file
        print(f"Reading input file: {args.input_file}")
        with open(args.input_file, "r", encoding=args.encoding) as file:
            content = file.read()

        # Process data
        results = process_parse_trees(content)

        # Save results
        print("Saving results...")
        save_results(results, args.output_file)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        return 1
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
