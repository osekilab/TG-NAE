"""Merge the per-condition, per-seed NAE files into a single predictor table.

The files under outputs/seed=*/ hold one condition each, and all of them are
aligned to the same list of Natural Stories tokens. This script joins them on
the token id so that the predictors can be read with a single call, and turns
the log probabilities into surprisals.

Usage:
    python ./src/merge_predictors.py \
        --outputs_dir ./outputs \
        --output_file ./outputs/nae_predictors.csv
"""

import argparse
import csv
import os

SEEDS = ["0", "123", "1234"]

# (file name without extension, whether the condition involves a syntactic stack)
CONDITIONS = [
    ("txl", False),
    ("tg_gold", True),
    ("tg_bs", True),
    ("txl_tree_gold", True),
    ("txl_tree_bs", True),
]


def read_condition(path):
    """Read one condition file, keyed by nothing but kept in file order"""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def negate(value):
    """Flip the sign of a decimal string without going through a float

    The log probabilities are non-positive, so the surprisal is obtained by
    dropping the leading minus sign. Doing this on the string keeps the values
    bit-for-bit identical to the ones in the per-condition files.
    """
    value = value.strip()
    if value.startswith("-"):
        return value[1:]
    if float(value) == 0.0:
        return value
    raise ValueError(f"expected a non-positive log probability, got {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--output_file", default="./outputs/nae_predictors.csv")
    args = parser.parse_args()

    # Read every condition and seed
    tables = {}
    for seed in SEEDS:
        for condition, _ in CONDITIONS:
            path = os.path.join(args.outputs_dir, f"seed={seed}", f"{condition}.csv")
            tables[(condition, seed)] = read_condition(path)

    # The files are aligned by construction; refuse to merge them if they are not
    reference = tables[(CONDITIONS[0][0], SEEDS[0])]
    for (condition, seed), rows in tables.items():
        if len(rows) != len(reference):
            raise ValueError(
                f"seed={seed}/{condition}.csv has {len(rows)} rows, "
                f"expected {len(reference)}"
            )
        for i, (row, ref) in enumerate(zip(rows, reference)):
            if row["id"] != ref["id"] or row["word"] != ref["word"]:
                raise ValueError(
                    f"seed={seed}/{condition}.csv diverges from the reference at "
                    f"row {i}: ({row['id']}, {row['word']}) "
                    f"vs ({ref['id']}, {ref['word']})"
                )

    # Build the header. Every metric also gets the mean over the seeds, taken on
    # the raw values; standardize after averaging, not before.
    fieldnames = ["id", "story", "zone", "word"]
    for condition, has_stack in CONDITIONS:
        metrics = ["surp", "nae", "stack"] if has_stack else ["surp", "nae"]
        for metric in metrics:
            for seed in SEEDS:
                fieldnames.append(f"{condition}_{metric}_seed{seed}")
            fieldnames.append(f"{condition}_{metric}_mean")

    with open(args.output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, ref in enumerate(reference):
            story, zone = ref["id"].split(".", 1)
            out = {
                "id": ref["id"],
                "story": story,
                "zone": zone,
                "word": ref["word"],
            }
            for condition, has_stack in CONDITIONS:
                for seed in SEEDS:
                    row = tables[(condition, seed)][i]
                    out[f"{condition}_surp_seed{seed}"] = negate(row["sum_log_prob"])
                    out[f"{condition}_nae_seed{seed}"] = row[
                        "sum_projected_metrics_nae"
                    ]
                    if has_stack:
                        out[f"{condition}_stack_seed{seed}"] = row["sum_stack_count"]

                metrics = ["surp", "nae", "stack"] if has_stack else ["surp", "nae"]
                for metric in metrics:
                    values = [
                        float(out[f"{condition}_{metric}_seed{seed}"])
                        for seed in SEEDS
                    ]
                    out[f"{condition}_{metric}_mean"] = repr(sum(values) / len(values))
            writer.writerow(out)

    print(f"Wrote {len(reference)} rows and {len(fieldnames)} columns to {args.output_file}")


if __name__ == "__main__":
    main()
