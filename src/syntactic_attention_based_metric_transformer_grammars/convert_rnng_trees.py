import argparse
import os
from collections import defaultdict


def complete_brackets(tree):
    stack = []
    for char in tree:
        if char == "(":
            stack.append(char)
        elif char == ")":
            stack.pop()

    return tree + ")" * len(stack)


def parse_input_text(text):
    current_sent = None
    current_pointer = None
    trees = defaultdict(lambda: defaultdict(list))

    for line in text.strip().split("\n"):
        if line.startswith("Sentence"):
            current_sent = int(line.split()[1].rstrip(":"))
        elif line.startswith("Pointer:"):
            current_pointer = int(line.split()[1])
        elif line.startswith("("):
            if current_sent is not None and current_pointer is not None:
                completed_tree = complete_brackets(line.strip())
                trees[current_sent][current_pointer].append(completed_tree)

    return trees


def save_trees(trees, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for sent_num, pointers in trees.items():
        sent_dir = os.path.join(output_dir, f"sent_{sent_num}")
        os.makedirs(sent_dir, exist_ok=True)

        for pointer_num, pointer_trees in pointers.items():
            pointer_file = os.path.join(sent_dir, f"pointer_{pointer_num}.txt")
            with open(pointer_file, "w") as f:
                for tree in pointer_trees:
                    f.write(tree + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse tree organizer")
    parser.add_argument("--input_file", help="Input file containing the parse trees")
    parser.add_argument("--output_dir", help="Output directory for organized trees")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        input_text = f.read()

    trees = parse_input_text(input_text)
    save_trees(trees, args.output_dir)

    print(f"Trees have been organized and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
