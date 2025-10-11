import argparse
from typing import List

import nltk
from tqdm import tqdm


def process_treebank_text(text: str) -> List[List[str]]:
    sentences = []
    current_sentence = ""
    parentheses_count = 0

    for char in tqdm(text):
        current_sentence += char
        if char == "(":
            parentheses_count += 1
        elif char == ")":
            parentheses_count -= 1
            if parentheses_count == 0 and current_sentence.strip():
                sentences.append(current_sentence.strip())
                current_sentence = ""

    results = []

    for sentence in tqdm(sentences):
        try:
            tree = nltk.Tree.fromstring(sentence)
            word_pos_pairs = tree.pos()
            words = [word for word, pos in word_pos_pairs if pos != "-NONE-"]

            results.append(words)

        except ValueError as e:
            print(f"Error parsing sentence: {e}")
            continue

    return results


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--natural_stories_ptb_path", type=str, help="Path to Penn Treebank file"
    )
    parse.add_argument(
        "--output_path", type=str, help="Path to save extracted terminals"
    )
    args = parse.parse_args()

    with open(args.natural_stories_ptb_path, "r") as f:
        text = f.read()

    results = process_treebank_text(text)

    with open(args.output_path, "w") as f:
        for result in results:
            f.write(" ".join(result) + "\n")


if __name__ == "__main__":
    main()
